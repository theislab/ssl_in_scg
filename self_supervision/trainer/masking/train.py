import argparse
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from self_supervision.data.checkpoint_utils import (
    load_last_checkpoint,
    checkpoint_exists,
)
from self_supervision.trainer.masking.mask_utils import (
    encode_gene_programs,
    encode_gene_program_to_transcription_factor,
    read_gmt_to_dict,
    read_gmt,
)
from self_supervision.estimator.cellnet import EstimatorAutoEncoder
from self_supervision.paths import DATA_DIR, TRAINING_FOLDER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", action="store_true", help="Whether to use decoder")
    parser.add_argument(
        "--model",
        type=str,
        default="MLP",
        choices=["MLP", "NegBin"],
        help="Model to use",
    )
    parser.add_argument("--mask_rate", type=float, default=None, help="Masking rate")
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default=None,
        choices=["random", "gene_program", "gp_to_tf", "single_gene_program", "None"],
        help="Masking strategy."
        "random - random masking"
        "gene_program - masking with gene programs"
        "gp_to_tf - masking with gene programs to transcription factors",
    )
    parser.add_argument(
        "--donor_list",
        type=str,
        default=None,
        help="Path to donor list file",
    )
    parser.add_argument(
        "--gp_file",
        type=str,
        default="C5",
        choices=["C3", "C5", "C8"],
        help="Path to gene program file. "
        "C3 collection - coupled gene programs and transcription factors. "
        "C5 collection - ontology gene sets, very general. "
        "C8 collection - cell type signature gene sets.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--batch_size", default=8192, type=int, help="Batch size")
    parser.add_argument("--mask_type", default="sparsemax", type=str, help="Mask type")
    parser.add_argument("--version", type=str, default="", help="Version of the model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hidden_units",
        type=list,
        default=[512, 512, 256, 256, 64],
        help="Hidden units",
    )
    parser.add_argument(
        "--checkpoint_interval", default=1, type=int, help="Checkpoint interval"
    )
    parser.add_argument(
        "--missing_tolerance", default=0, type=int, help="Missing tolerance"
    )
    parser.add_argument(
        "--data_perc",
        default=1e0,
        type=float,
        help="Percentage of data to use for training",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    print(args)

    # FIX SEED FOR REPRODUCIBILITY
    # seed_everything(90)
    torch.manual_seed(0)

    # if args.mask_rate is not None but args.masking_strategy is None, args.masking_strategy is set to 'random'
    if args.mask_rate is not None and args.masking_strategy is None:
        args.masking_strategy = "random"

    # CHECKPOINT HANDLING
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    print("Root: ", root)
    if args.masking_strategy == "random":
        subfolder = (
            args.model
            + "_"
            + str(int(100 * args.mask_rate))
            + "p"
            + args.version
            + "/"
        )
    elif args.masking_strategy == "gene_program":
        subfolder = (
            args.model
            + "_"
            + args.masking_strategy
            + "_"
            + args.gp_file
            + "_"
            + str(int(100 * args.mask_rate))
            + "p"
            + args.version
            + "/"
        )
    elif (
        args.masking_strategy == "gp_to_tf"
        or args.masking_strategy == "single_gene_program"
    ):
        subfolder = (
            args.model + "_" + args.masking_strategy + args.version + "/"
        )
    elif args.masking_strategy is None:
        subfolder = args.model + "_" + "no_mask" + args.version + "/"
    else:
        raise ValueError(
            "args.masking_strategy needs to be random, gene_program, or None"
        )
    CHECKPOINT_PATH = os.path.join(
        TRAINING_FOLDER,
        "pretext_models",
        "masking",
        "CN_" + subfolder,
    )
    print("Will save model to " + CHECKPOINT_PATH)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    estim = EstimatorAutoEncoder(
        data_path=os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p"),
    )

    # set up datamodule
    estim.init_datamodule(batch_size=args.batch_size, sub_sample_frac=args.data_perc)

    estim.init_trainer(
        trainer_kwargs={
            "max_epochs": 1000,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "default_root_dir": CHECKPOINT_PATH,
            "accelerator": "gpu",
            "devices": 1,
            "num_sanity_val_steps": 0,
            "logger": [TensorBoardLogger(CHECKPOINT_PATH, name="default")],
            "detect_anomaly": False,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "enable_checkpointing": True,
            "callbacks": [
                TQDMProgressBar(refresh_rate=300),
                LearningRateMonitor(logging_interval="step"),
                # Save the model with the best training loss
                ModelCheckpoint(
                    filename="best_checkpoint_train",
                    monitor="train_loss_epoch",
                    mode="min",
                    every_n_epochs=args.checkpoint_interval,
                    save_top_k=1,
                ),
                # Save the model with the best validation loss
                ModelCheckpoint(
                    filename="best_checkpoint_val",
                    monitor="val_loss",
                    mode="min",
                    every_n_epochs=args.checkpoint_interval,
                    save_top_k=1,
                ),
                ModelCheckpoint(filename="last_checkpoint", monitor=None),
            ],
        }
    )

    # get gene program
    if args.masking_strategy == "random" or args.masking_strategy is None:
        encoded_gene_program = None
    elif (
        args.masking_strategy == "gene_program"
        or args.masking_strategy == "single_gene_program"
    ):
        var_names = np.array(
            pd.read_parquet(os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p", "var.parquet"))[
                "feature_name"
            ].tolist()
        )

        if args.gp_file == "C5":  # C5 collection - ontology gene sets, very general
            gp_file = (
                "/self_supervision/data/gene_programs/c5.go.v2022.1.Hs.symbols.gmt"
            )
        elif (
            args.gp_file == "C8"
        ):  # Choose the C8 collection - cell type signature gene sets
            gp_file = (
                "/self_supervision/data/gene_programs/c8.all.v2023.1.Hs.symbols.gmt"
            )
        elif (
            args.gp_file == "C3"
        ):  # Choose the C3 collection - coupled gene programs and transcription factors
            gp_file = (
                "/self_supervision/data/gene_programs/c3.tft.v2023.1.Hs.symbols.gmt"
            )
        else:
            raise ValueError("args.gp_file needs to be C3, C5 or C8")
        gene_program = read_gmt(gp_file=root + gp_file)
        with torch.no_grad():
            encoded_gene_program = encode_gene_programs(
                var_names=var_names,
                gene_program=gene_program,
                required_tolerance=args.missing_tolerance,
            )
    elif args.masking_strategy == "gp_to_tf":
        gp_file = "/self_supervision/data/gene_programs/c3.tft.v2023.1.Hs.symbols.gmt"
        var_names = np.array(
            pd.read_parquet(os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p", "var.parquet"))[
                "feature_name"
            ].tolist()
        )
        gene_program = read_gmt_to_dict(gp_file=root + gp_file)
        with torch.no_grad():
            # caution, encoded_gene_program is a dict of tuples of tensors
            encoded_gene_program = encode_gene_program_to_transcription_factor(
                var_names=var_names,
                gene_program=gene_program,
                required_tolerance=args.missing_tolerance,
            )
    else:
        raise ValueError(
            "args.masking_strategy needs to be random, gene_program, or None"
        )

    # init model
    # reproducibility
    if args.model == "MLP":
        model_type = "mlp_ae"
    elif args.model == "NegBin":
        model_type = "mlp_negbin"

    estim.init_model(
        model_type=model_type,
        model_kwargs={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
            "masking_strategy": args.masking_strategy,
            "masking_rate": args.mask_rate,
            "encoded_gene_program": encoded_gene_program,
            "units_encoder": args.hidden_units,
            "units_decoder": args.hidden_units[::-1][1:] if args.decoder else [],
            "donor_subset": np.load(args.donor_list) if args.donor_list else None,
        },
    )

    # check if there are .ckpt files in the checkpoint directory
    if checkpoint_exists(CHECKPOINT_PATH):
        print(
            "No loading of pre-trained weights, continue training from", CHECKPOINT_PATH
        )
        ckpt = load_last_checkpoint(CHECKPOINT_PATH)
        print("Load checkpoint from", ckpt)
        estim.train(ckpt_path=ckpt)
    else:
        print("No loading of pre-trained weights, start training from scratch")
        estim.train()


if __name__ == "__main__":
    train()
