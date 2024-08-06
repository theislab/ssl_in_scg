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
from self_supervision.paths import DATA_DIR, TRAINING_FOLDER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=dict, default=None)
    parser.add_argument("--contrastive_method", type=str, default="BYOL")
    parser.add_argument("--p", type=float, default=0.5, help="Likelihood of data augmentation")
    parser.add_argument("--negbin_intensity", type=float, default=0.1, help="Negative binomial intensity")
    parser.add_argument("--dropout_intensity", type=float, default=0.1, help="Dropout intensity")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--learning_rate_weights", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden_units", type=list, default=[512, 512, 256, 256, 64])
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--max_epochs", type=int, default=1000)
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
    subfolder = (
        args.contrastive_method
        + "_p_"
        + str(args.p)
        + "_negbin_"
        + str(args.negbin_intensity)
        + "_dropout_"
        + str(args.dropout_intensity)
        + "_lr_"
        + str(args.lr)
        + "_wd_"
        + str(args.weight_decay)
        + str(args.version)
    )
    if "." in subfolder:  # replace dot with underscore
        subfolder = subfolder.replace(".", "_")
    # remove the "_" at the end of the string if it is there
    if subfolder[-1] == "_":
        subfolder = subfolder[:-1]
    CHECKPOINT_PATH = os.path.join(
        TRAINING_FOLDER, "pretext_models", "contrastive", subfolder
    )
    print("Will save model to " + CHECKPOINT_PATH)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    # get estimator
    estim = EstimatorAutoEncoder(        
        data_path=os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p"),
    )

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
                "units_encoder": args.hidden_units,
                "negbin_intensity": args.negbin_intensity,
                "dropout_intensity": args.dropout_intensity,
                "lr": args.lr,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "CHECKPOINT_PATH": CHECKPOINT_PATH,
            },
        )
        print("Training Barlow Twins")
        print("Saving model to " + CHECKPOINT_PATH)
        estim.train()
    else:
        estim.init_trainer(
            trainer_kwargs={
                "max_epochs": args.max_epochs,
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
                "units_encoder": args.hidden_units,
                "p": args.p,
                "negbin_intensity": args.negbin_intensity,
                "dropout_intensity": args.dropout_intensity,
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
