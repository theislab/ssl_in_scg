import scanpy as sc
import lightning.pytorch as pl
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
import os
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import pandas as pd
from pathlib import Path
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    LearningRateMonitor,
    ModelCheckpoint,
)
from self_supervision.data.datamodules import MerlinDataModule, MultiomicsDataloader
from self_supervision.models.lightning_modules.multiomics_autoencoder import (
    MultiomicsAutoencoder,
    MLPAutoEncoder,
    FG_BG_MultiomicsAutoencoder,
)
from self_supervision.models.lightning_modules.multiomics_autoencoder import (
    MLPBarlowTwins,
    MLPBYOL,
)
from self_supervision.data.checkpoint_utils import (
    load_last_checkpoint,
    checkpoint_exists,
)
from self_supervision.trainer.multiomics.multiomics_utils import (
    one_hot_encode,
    filter_gene_set,
    encode_tf,
    read_gmt_to_dict,
    encode_gene_program,
)
from collections import OrderedDict


def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
    - argparse.Namespace containing all the hyperparameters and settings.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", default=0.11642113240634665, type=float)
    parser.add_argument("--weight_decay", default=0.0010851761758488817, type=float)
    parser.add_argument("--learning_rate", default=0.00011197711341004587, type=float)
    parser.add_argument(
        "--mode",
        default="pre_training",
        type=str,
        help="choose from pre_training, fine_tuning",
    )
    parser.add_argument(
        "--model", default="MAE", type=str, help="choose from MAE, BYOL, BT, NegBin"
    )
    parser.add_argument("--batch_size", default=16384, type=int)
    parser.add_argument(
        "--dataset", default="NeurIPS", type=str, help="choose from NeurIPS, 20M"
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        help="choose from random,gene_program,gp_to_tf, gp_to_gp",
    )
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument(
        "--model_type", default="autoencoder", type=str, choices=["autoencoder", "vae"]
    )
    parser.add_argument("--pretrained_dir", default=None, type=str)
    parser.add_argument(
        "--model_path",
        default="/lustre/groups/ml01/workspace/till.richter/",
        type=str,
        help="Path where the lightning checkpoints are stored",
    )
    parser.add_argument(
        "--data_dir",
        default="/lustre/groups/ml01/workspace/till.richter/",
        type=str,
        help="Path where the data is stored",
    )
    parser.add_argument("--version", default="", type=str)

    return parser.parse_args()


def train(
    args: argparse.Namespace,
    train_dl: DataLoader,
    val_dl: DataLoader,
    tf_program: Optional[dict] = None,
    encoded_genes: Optional[dict] = None,
    datamodule: Optional[MerlinDataModule] = None,
) -> float:
    """
    Train function for an autoencoder or variational autoencoder model.

    Parameters:
    - args: argparse.Namespace, containing all the hyperparameters and settings.
    """

    # CHECKPOINT HANDLING
    if args.mode == "pre_training":
        # if any of the args.xy is None, make args.xy -> 'None' for the subfolder to have a name
        if args.masking_strategy is None:
            args.masking_strategy = "None"
        if args.model is None:
            args.model = "None"
        if args.version is None:
            args.version = "None"

        subfolder = (
            "multiomics_"
            + args.dataset
            + "_"
            + args.masking_strategy
            + "_"
            + args.model
            + "_"
            + args.version
        )
        CHECKPOINT_PATH = os.path.join(
            args.model_path, "trained_models", "pretext_models", "multiomics", subfolder
        )

    elif args.mode == "fine_tuning":
        if args.pretrained_dir:
            if "reconstruction" in args.pretrained_dir:
                pre_model = (
                    args.pretrained_dir.split("/")[-6]
                    + "_"
                    + args.pretrained_dir.split("/")[-5]
                )
            # Barlow Twins NeurIPS Data
            elif (
                "multiomics_NeurIPS_None_BT_bt_pretrain_neurips" in args.pretrained_dir
            ):
                pre_model = "BT_NeurIPS"
            # Barlow Twins scTab Data
            elif "multiomics_20M_None_BT_bt_pretrain_20m" in args.pretrained_dir:
                pre_model = "BT_20M"
            # BYOL NeurIPS Data
            elif (
                "multiomics_NeurIPS_None_BYOL_bt_pretrain_neurips"
                in args.pretrained_dir
            ):
                pre_model = "BYOL_NeurIPS"
            # BYOL scTab Data
            elif "multiomics_20M_None_BYOL_bt_pretrain_20m" in args.pretrained_dir:
                pre_model = "BYOL_20M"
            # GP to GP NeurIPS Data
            elif (
                "multiomics_NeurIPS_gp_to_gp_MAE_GP_to_GP_Pretrain"
                in args.pretrained_dir
            ):
                pre_model = "GP_to_GP_NeurIPS"
            # GP to GP scTab Data
            elif (
                "multiomics_20M_gp_to_gp_MAE_gp2gp_Pretrain_20M_v1"
                in args.pretrained_dir
            ):
                pre_model = "GP_to_GP_20M"
            # GP to TF NeurIPS Data
            elif (
                "multiomics_NeurIPS_gp_to_gp_MAE_GP_to_GP_Pretrain"
                in args.pretrained_dir
            ):
                pre_model = "GP_to_GP_NeurIPS"
            # GP to TF scTab Data
            elif "multiomics_20M_gp_to_tf_MAE_v1" in args.pretrained_dir:
                pre_model = "GP_to_TF_20M"
            # Random Mask NeurIPS Data
            elif (
                "multiomics_NeurIPS_random_MAE_Ind_Mask_Pretrain" in args.pretrained_dir
            ):
                pre_model = "Random_Mask_NeurIPS"
            # Random Mask scTab Data
            elif (
                "multiomics_20M_random_MAE_Ind_Mask_Pretrain_20M_v1_Test2a"
                in args.pretrained_dir
            ):
                pre_model = "Random_Mask_20M"
            # Gene Program Mask NeurIPS Data
            elif "multiomics_NeurIPS_gene_program_MAE_" in args.pretrained_dir:
                pre_model = "Gene_Program_Mask_NeurIPS"
            # Gene Program Mask scTab Data
            # TO DO

            else:
                pre_model = args.pretrained_dir.split("/")[-5]
        else:
            pre_model = None

        if pre_model:
            subfolder = "SSL_" + pre_model + args.version
        else:
            subfolder = "No_SSL_" + args.version

        if args.model == "NegBin":
            subfolder = "NegBin_" + subfolder

        CHECKPOINT_PATH = os.path.join(
            args.model_path, "trained_models", "final_models", "multiomics", subfolder
        )

    else:
        raise ValueError("Mode {} not supported".format(args.mode))

    print("Will save model to " + CHECKPOINT_PATH)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    # Initialize the model
    print("Initialize the model...")
    if args.mode == "pre_training" and args.masking_strategy == "gene_program":
        if args.mode == "pre_training" and args.masking_strategy == "random":
            ae_model = MLPAutoEncoder(
                masking_strategy=args.masking_strategy,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                n_data=90261 if args.dataset == "NeurIPS" else sum(datamodule.train_dataset.partition_lens)
            )

        elif args.mode == "pre_training" and (args.masking_strategy == "gp_to_tf" or args.masking_strategy == "gp_to_gp"):
            ae_model = MultiomicsAutoencoder(
                mode=args.mode,
                model=args.model,
                masking_strategy=args.masking_strategy,
                encoded_gene_program=tf_program,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                n_data=90261 if args.dataset == "NeurIPS" else sum(datamodule.train_dataset.partition_lens),
                model_type=args.model_type  # Standard is 'autoencoder'
            )

        elif args.mode == "pre_training" and args.model == "BT":
            ae_model = MLPBarlowTwins(
                gene_dim=2000,
                units_encoder=[256, 256, 40],
                train_set_size=90000 // 256 if args.dataset == "NeurIPS" else sum(datamodule.train_dataset.partition_lens),
                val_set_size=90000 // 256 if args.dataset == "NeurIPS" else sum(datamodule.val_dataset.partition_lens),
                batch_size=args.batch_size,
                hvg=False,
                num_hvgs=2000,
                CHECKPOINT_PATH=CHECKPOINT_PATH,
                backbone="MLP",  # MLP, TabNet
                augment_intensity=0.001,
                activation=nn.ReLU
            )

        elif args.mode == "pre_training" and args.model == "BYOL":
            ae_model = MLPBYOL(
                gene_dim=2000,
                units_encoder=[256, 256, 40],
                train_set_size=90000 // 256 if args.dataset == "NeurIPS" else sum(datamodule.train_dataset.partition_lens),
                val_set_size=90000 // 256 if args.dataset == "NeurIPS" else sum(datamodule.val_dataset.partition_lens),
                batch_size=args.batch_size,
                hvg=False,
                num_hvgs=2000,
                backbone="MLP",  # MLP, TabNet
                augment_intensity=0.001,
                augment_type="Gaussian",
                use_momentum=True,
                activation=nn.ReLU
            )

        elif args.mode == "fine_tuning" and args.model == "MAE":
            ae_model = MultiomicsAutoencoder(
                mode=args.mode,
                model=args.model,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                model_type=args.model_type  # Standard is 'autoencoder'
            )

        elif args.mode == "fine_tuning" and args.model == "NegBin":
            ae_model = FG_BG_MultiomicsAutoencoder(
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size
            )

        else:
            raise ValueError("Model {} not supported".format(args.model))

        # Load pretrained weights
        if args.mode == "fine_tuning" and args.pretrained_dir is not None:
            print("Load pretrained weights...")
            checkpoint = torch.load(args.pretrained_dir)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                pretrained_dict = {
                    k: v
                    for k, v in checkpoint["state_dict"].items()
                    if ("decoder" not in k) and ("projector" not in k)
                }
            elif isinstance(checkpoint, OrderedDict):
                pretrained_dict = {
                    k: v
                    for k, v in checkpoint.items()
                    if ("decoder" not in k) and ("projector" not in k)
                }
            ae_model.load_state_dict(pretrained_dict, strict=False)

        # Initialize the trainer
        trainer_kwargs = {
            "max_epochs": 5000,
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
                    every_n_epochs=1,
                    save_top_k=1,
                ),
                # Save the model with the best validation loss
                ModelCheckpoint(
                    filename="best_checkpoint_val",
                    monitor="val_loss",
                    mode="min",
                    every_n_epochs=1,
                    save_top_k=1,
                ),
                ModelCheckpoint(filename="last_checkpoint", monitor=None),
            ],
        }

    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    assert isinstance(ae_model, pl.LightningModule)

    # check if there are .ckpt files in the checkpoint directory
    if checkpoint_exists(CHECKPOINT_PATH):
        print(
            "No loading of pre-trained weights, continue training from", CHECKPOINT_PATH
        )
        # get best checkpoint from previous training with PyTorch Lightning
        # ckpt = load_best_checkpoint(CHECKPOINT_PATH)
        ckpt = load_last_checkpoint(CHECKPOINT_PATH)
        print("Load checkpoint from", ckpt)
        trainer.fit(
            ae_model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt
        )
    else:
        print("No loading of pre-trained weights, start training from scratch")
        trainer.fit(ae_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Retrieve the best validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss


if __name__ == "__main__":
    args = parse_arguments()
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    print("Root directory is " + root)

    tf_program = None
    encoded_genes = None

    if args.dataset == "NeurIPS":
        # Data loading and preprocessing
        adata = sc.read_h5ad(
            os.path.join(args.data_dir, "NeurIPS_filtered_hvg_adata.h5ad")
        )
        adata.X = adata.layers["counts"].copy()

        if args.mode == "pre_training" and args.masking_strategy == "gene_program":
            print("preprocessing gene program...")
            gene_set = pd.read_csv(
                os.path.join(args.data_dir, "gene_program.csv"), index_col=0
            )  # adjust filename

            gene_set[["overlap_num", "total_num"]] = gene_set["Overlap"].str.split(
                "/", expand=True
            )
            gene_set["total_num"] = gene_set["total_num"].astype(int)
            gene_set["overlap_num"] = gene_set["overlap_num"].astype(int)
            gene_set["overlap_pct"] = gene_set["overlap_num"] / gene_set["total_num"]

            filtered_gene_set = filter_gene_set(gene_set)

            overlapped_genes = filtered_gene_set.Genes.str.split(";").values
            encoded_genes = encode_gene_program(overlapped_genes, adata)

            print("Encoded genes: ", encoded_genes)

        elif args.mode == "pre_training" and (
            args.masking_strategy == "gp_to_tf" or args.masking_strategy == "gp_to_gp"
        ):
            print("preprocessing gp_to_tf/gp_to_gp...")
            gp_file = (
                "/self_supervision/data/gene_programs/c3.tft.v2023.1.Hs.symbols.gmt"
            )
            print("Reading gene program file from " + root + gp_file)
            tf_dict = read_gmt_to_dict(root + gp_file)

            tf_program = encode_tf(adata.var_names.to_list(), tf_dict)

        # Split data based on the newly defined 'split' attribute
        train_ds = MultiomicsDataloader(
            np.log1p(adata[adata.obs["split"] == "train"].obsm["protein_counts"]),
            adata[adata.obs["split"] == "train"].X.todense(),
            one_hot_encode(adata[adata.obs["split"] == "train"].obs["batch"]),
        )

        val_ds = MultiomicsDataloader(
            np.log1p(adata[adata.obs["split"] == "val"].obsm["protein_counts"]),
            adata[adata.obs["split"] == "val"].X.todense(),
            one_hot_encode(adata[adata.obs["split"] == "val"].obs["batch"]),
        )

        test_ds = MultiomicsDataloader(
            np.log1p(adata[adata.obs["split"] == "test"].obsm["protein_counts"]),
            adata[adata.obs["split"] == "test"].X.todense(),
            one_hot_encode(adata[adata.obs["split"] == "test"].obs["batch"]),
        )

        ood_test_ds = MultiomicsDataloader(
            np.log1p(adata[adata.obs["split"] == "ood_test"].obsm["protein_counts"]),
            adata[adata.obs["split"] == "ood_test"].X.todense(),
            one_hot_encode(adata[adata.obs["split"] == "ood_test"].obs["batch"]),
        )

        # Create DataLoaders
        cpus = os.cpu_count()
        num_workers = max(1, cpus // 2)  # Ensure at least one worker is used

        train_dl = DataLoader(
            train_ds, shuffle=True, batch_size=args.batch_size, num_workers=num_workers
        )
        val_dl = DataLoader(
            val_ds, shuffle=False, batch_size=args.batch_size, num_workers=num_workers
        )
        test_dl = DataLoader(
            test_ds, shuffle=False, batch_size=args.batch_size, num_workers=num_workers
        )  # Updated batch_size from 1 to args.batch_size
        ood_test_dl = DataLoader(
            ood_test_ds,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )  # New DataLoader for OOD test set
        datamodule = None

    else:
        if args.mode == "pre_training" and args.masking_strategy == "gene_program":
            # Data loading and preprocessing
            adata = sc.read_h5ad(
                os.path.join(args.data_dir, "NeurIPS_filtered_hvg_adata.h5ad")
            )
            adata.X = adata.layers["counts"].copy()

            print("preprocessing gene program...")
            gene_set = pd.read_csv(
                os.path.join(args.data_dir, "gene_program.csv"), index_col=0
            )  # adjust filename

            gene_set[["overlap_num", "total_num"]] = gene_set["Overlap"].str.split(
                "/", expand=True
            )
            gene_set["total_num"] = gene_set["total_num"].astype(int)
            gene_set["overlap_num"] = gene_set["overlap_num"].astype(int)
            gene_set["overlap_pct"] = gene_set["overlap_num"] / gene_set["total_num"]

            filtered_gene_set = filter_gene_set(gene_set)

            overlapped_genes = filtered_gene_set.Genes.str.split(";").values
            encoded_genes = encode_gene_program(overlapped_genes, adata)

        elif args.mode == "pre_training" and (
            args.masking_strategy == "gp_to_tf" or args.masking_strategy == "gp_to_gp"
        ):
            # Data loading and preprocessing
            adata = sc.read_h5ad(
                os.path.join(args.data_dir, "NeurIPS_filtered_hvg_adata.h5ad")
            )
            adata.X = adata.layers["counts"].copy()
            print("preprocessing gp_to_tf/gp_to_gp...")
            gp_file = (
                "/self_supervision/data/gene_programs/c3.tft.v2023.1.Hs.symbols.gmt"
            )
            print("Reading gene program file from " + root + gp_file)
            tf_dict = read_gmt_to_dict(root + gp_file)

            tf_program = encode_tf(adata.var_names.to_list(), tf_dict)

        datamodule = MerlinDataModule(
            path=os.path.join(args.data_dir, "merlin_cxg_2023_05_15_sf-log1p"),
            columns=["dataset_id"],
            batch_size=args.batch_size,
        )

        print("Train set size: ", sum(datamodule.train_dataset.partition_lens))

        train_dl = datamodule.train_dataloader()
        val_dl = datamodule.val_dataloader()

    # Training
    result = train(
        args=args,
        train_dl=train_dl,
        val_dl=val_dl,
        tf_program=tf_program,
        encoded_genes=encoded_genes,
        datamodule=datamodule,
    )
    print(result)
