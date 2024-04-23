import argparse
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import os
from pathlib import Path
from self_supervision.estimator.cellnet import EstimatorAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=dict, default=None)
    parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--contrastive_method", type=str, default="BYOL")
    parser.add_argument("--augment_type", type=str, default="Gaussian")
    parser.add_argument("--augment_intensity", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--learning_rate_weights", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden_units", type=list, default=[512, 512, 256, 256, 64])
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument("--version", type=str, default="")
    parser.add_argument(
        "--hvg", action="store_true", help="Whether to use highly variable genes"
    )
    parser.add_argument(
        "--num_hvgs",
        default=2000,
        type=int,
        help="Number of highly variable genes to use",
    )
    parser.add_argument(
        "--data_path",
        default="/lustre/groups/ml01/workspace/till.richter/merlin_cxg_2023_05_15_sf-log1p",
        type=str,
        help="Path to the data stored as parquet files",
    )
    # Old, 10M dataset: '/lustre/scratch/users/till.richter/merlin_cxg_simple_norm_parquet'
    parser.add_argument(
        "--model_path",
        default="/lustre/groups/ml01/workspace/till.richter/",
        type=str,
        help="Path where the lightning checkpoints are stored",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # set up environment
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()
    print(args)

    # FIX SEED FOR REPRODUCIBILITY
    torch.manual_seed(0)

    # CHECKPOINT HANDLING
    if args.hvg:
        use_hvg = "HVG_" + str(args.num_hvgs) + "_"
    else:
        use_hvg = ""
    subfolder = (
        use_hvg
        + args.model
        + "_"
        + args.contrastive_method
        + "_"
        + str(args.augment_type)
        + "_"
        + str(args.augment_intensity)
        + "_"
        + args.version
    )
    if "." in subfolder:  # replace dot with underscore
        subfolder = subfolder.replace(".", "_")
    # remove the "_" at the end of the string if it is there
    if subfolder[-1] == "_":
        subfolder = subfolder[:-1]
    CHECKPOINT_PATH = os.path.join(
        args.model_path, "trained_models", "pretext_models", "contrastive", subfolder
    )
    print("Will save model to " + CHECKPOINT_PATH)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    # get estimator
    estim = EstimatorAutoEncoder(data_path=args.data_path, hvg=args.hvg)

    # set up datamodule
    estim.init_datamodule(batch_size=args.batch_size)

    # INIT TRAINER
    if (
        args.contrastive_method == "bt"
    ):  # BarlowTwins has manual optimization, hence no gradient clipping
        estim.init_trainer(
            trainer_kwargs={
                "max_epochs": 1000,
                "gradient_clip_algorithm": "norm",
                "default_root_dir": CHECKPOINT_PATH,
                "accelerator": "gpu",
                "devices": 1,
                "num_sanity_val_steps": 0,
                "check_val_every_n_epoch": 1,
                "logger": [TensorBoardLogger(CHECKPOINT_PATH, name="default")],
                "log_every_n_steps": 100,
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
                    ),
                    # Save the model with the best validation loss
                    ModelCheckpoint(
                        filename="best_checkpoint_val", monitor="val_loss", mode="min"
                    ),
                    ModelCheckpoint(filename="last_checkpoint", monitor=None),
                ],
            }
        )
        estim.init_model(
            model_type="mlp_bt",
            model_kwargs={
                "backbone": "MLP",
                "units_encoder": [512, 512, 256, 256, 64],
                "learning_rate_weights": args.learning_rate_weights,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "augment_intensity": args.augment_intensity,
                "CHECKPOINT_PATH": CHECKPOINT_PATH,
                "train_set_size": sum(estim.datamodule.train_dataset.partition_lens)
            },
        )
        print("Training Barlow Twins")
        print("Saving model to " + CHECKPOINT_PATH)
        estim.train()
    else:
        estim.init_trainer(
            trainer_kwargs={
                "max_epochs": 1000,
                "gradient_clip_val": 1.0,
                "gradient_clip_algorithm": "norm",
                "default_root_dir": CHECKPOINT_PATH,
                # 'dirpath': CHECKPOINT_PATH,
                "accelerator": "gpu",
                "devices": 1,
                "num_sanity_val_steps": 0,
                "check_val_every_n_epoch": 1,
                "logger": [TensorBoardLogger(CHECKPOINT_PATH, name="default")],
                "log_every_n_steps": 100,
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
        # INIT MODEL
        estim.init_model(
            model_type="mlp_byol",
            model_kwargs={
                "backbone": args.model,
                "units_encoder": args.hidden_units,
                "augment_type": args.augment_type,
                "augment_intensity": args.augment_intensity,
                "lr": args.lr,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "use_momentum": args.contrastive_method == "BYOL"
                if args.contrastive_method in ("BYOL", "SimSiam")
                else (
                    "SimSiam"
                    if args.contrastive_method == "SimSiam"
                    else ValueError("Invalid contrastive method")
                ),
            },
        )
        estim.train()
