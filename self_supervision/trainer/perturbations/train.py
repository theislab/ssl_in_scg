import scanpy as sc
import argparse
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    LearningRateMonitor,
    ModelCheckpoint,
)
import lightning.pytorch as pl
from self_supervision.data.checkpoint_utils import (
    load_last_checkpoint,
    checkpoint_exists,
)
from self_supervision.data.datamodules import AdataDataset
from self_supervision.models.lightning_modules.cellnet_autoencoder import (
    MLPAutoEncoder,
    PertClassifier,
)


def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
    - argparse.Namespace containing all the hyperparameters and settings.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata_dir",
        type=str,
        default="/lustre/groups/ml01/workspace/till.richter/",
        help="Path to the h5ad file containing the adata object.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Default",
        help="Adata Split",
        choices=["Default", "MedSplit", "HardSplit"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Supervised",
        help="Type of model to train.",
        choices=["Supervised", "Unsupervised"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training."
    )
    parser.add_argument(
        "--pretrained_dir", type=str, default=None, help="Path to pretrained model."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/lustre/groups/ml01/workspace/till.richter/trained_models/final_models/perturbations/",
        help="Path to the directory where the checkpoints are stored.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/icb/till.richter/git/self_supervision/self_supervision/data/perturbations",
        help="Path to the directory where the perturbation files are stored.",
    )
    parser.add_argument(
        "--version", type=str, default="run0", help="Version of the model."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="Weight decay."
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    return parser.parse_args()


def clf_update_weights(pretrained_dir, model):
    """
    Update the weights of the model_dict based on possible matches in the pretrained dict
    Correct for some diverging layer-names

    Args:
        pretrained_dir: Directory of the pretrained-weights
        model: Model object

    Returns:
        Dictionary of final model weights, including pretrained and randomly initialized weights
    """
    # load and init dicts
    try:
        pretrained_dict = torch.load(pretrained_dir)["state_dict"]
    except KeyError:  # for manual saving, it's directly the state dict
        pretrained_dict = torch.load(pretrained_dir)
    model_dict = model.state_dict()
    print("dict keys: ", model_dict.keys(), pretrained_dict.keys())
    final_dict = dict()

    # get pre_key from pretrained-model
    if "masking" in pretrained_dir or "reconstruction" in pretrained_dir:
        prefix_in_pre_key = "encoder"
    elif "BYOL" in pretrained_dir or "SimSiam" in pretrained_dir:
        prefix_in_pre_key = "inner_model"
    elif "bt" in pretrained_dir:
        prefix_in_pre_key = "backbone"
    else:
        raise ValueError(
            "In pretrained dir {} I could not find ['masking', 'BYOL', 'SimSiam']".format(
                pretrained_dir
            )
        )

    # iterate over all keys in model_dict
    for final_key in model_dict.keys():
        match_found = False
        # iterate over all keys in pretrained_dict
        for pre_key in pretrained_dict.keys():
            # check if key is in pretrained_dict
            if pre_key == final_key:
                print("Weight transfer: Direct match at", pre_key)
                # update final_dict with pretrained_dict
                final_dict.update({pre_key: pretrained_dict[pre_key]})
                # set match_found to True and break
                match_found = True
                break
            # check if key is in pretrained_dict
            elif prefix_in_pre_key in pre_key:
                prefix = "classifier"
                # get suffix of pre_key
                suffix = pre_key.split(prefix_in_pre_key)[1]
                # print('prefix + suffix: ', prefix + suffix, '\nfinal_key: ', final_key, '\npre_key: ', pre_key, '\n')
                if prefix + suffix == final_key:
                    print("Weight transfer: Matched", prefix + suffix)
                    # update final_dict with pretrained_dict
                    final_dict.update({prefix + suffix: pretrained_dict[pre_key]})
                    match_found = True
                    break

        if not match_found:
            print(
                "Did not find match for",
                final_key,
                "in pretrained_dict. Initializing randomly.",
            )
            final_dict.update({final_key: model_dict[final_key]})

    return final_dict


def update_weights(
    pretrained_dir: str, model: torch.nn.Module, model_type: str = "NegBin"
):
    """
    Update the weights of the model_dict based on possible matches in the pretrained dict
    Correct for some diverging layer-names

    Args:
        pretrained_dir: Directory of the pretrained-weights
        model: Model object
        model_type: Type of model to train.

    Returns:
        Dictionary of final model weights, including pretrained and randomly initialized weights
    """
    # load and init dicts
    try:
        pretrained_dict = torch.load(pretrained_dir)["state_dict"]
    except KeyError:  # for manual saving, it's directly the state dict
        pretrained_dict = torch.load(pretrained_dir)
    model_dict = model.state_dict()
    print("dict keys: ", model_dict.keys(), pretrained_dict.keys())
    final_dict = dict()

    # get pre_key from pretrained-model
    if "masking" in pretrained_dir:
        prefix_in_pre_key = "encoder"
    elif "BYOL" in pretrained_dir or "SimSiam" in pretrained_dir:
        prefix_in_pre_key = "inner_model"
    elif "classification" in pretrained_dir:
        prefix_in_pre_key = "classifier"
    elif "bt" in pretrained_dir:
        prefix_in_pre_key = "backbone"
    else:
        raise ValueError(
            "In pretrained dir {} I could not find ['masking', 'BYOL', 'SimSiam', 'classification']".format(
                pretrained_dir
            )
        )

    # iterate over all keys in model_dict
    for final_key in model_dict.keys():
        match_found = False
        # manual exception if pretrained dir is gp_to_tf, then ignore decoder.0.0.weight and decoder.0.0.bias
        if "gp_to_tf" in pretrained_dir and "decoder.0.0" in final_key:
            print(
                "Manual exception for GP TF. Did not find match for",
                final_key,
                "in pretrained_dict. Initializing randomly.",
            )
            final_dict.update({final_key: model_dict[final_key]})
        # iterate over all keys in pretrained_dict
        for pre_key in pretrained_dict.keys():
            if (model_type == "NegBin" and pre_key == "decoder.0.12.weight") or (
                model_type == "NegBin" and pre_key == "decoder.0.12.bias"
            ):
                print(
                    "Manual exception for decoder.0.12.weight. Initializing randomly."
                )
                final_dict.update({final_key: model_dict[final_key]})
                continue
            # check if key is in pretrained_dict
            if pre_key == final_key:
                print("weight transfer: Direct match at", pre_key)
                # update final_dict with pretrained_dict
                final_dict.update({pre_key: pretrained_dict[pre_key]})
                # set match_found to True and break
                match_found = True
                break
            # check if key is in pretrained_dict
            elif prefix_in_pre_key in pre_key:
                prefix = "encoder"
                # get suffix of pre_key
                suffix = pre_key.split(prefix_in_pre_key)[1]
                if prefix + suffix == final_key:
                    print("Weight transfer: Matched", prefix + suffix)
                    # update final_dict with pretrained_dict
                    final_dict.update({prefix + suffix: pretrained_dict[pre_key]})
                    match_found = True
                    break
        if not match_found:
            print(
                "Did not find match for",
                final_key,
                "in pretrained_dict. Initializing randomly.",
            )
            final_dict.update({final_key: model_dict[final_key]})

    return final_dict


def train(
    args: argparse.Namespace,
    CHECKPOINT_PATH: str,
    model,
    train_dl: DataLoader,
    val_dl: DataLoader,
):
    """
    Train the model.

    Args:
    - args: argparse.Namespace containing all the hyperparameters and settings.
    - CHECKPOINT_PATH: Path to the directory where the checkpoints are stored.
    - model: The model.
    - train_dl: The training dataloader.
    - val_dl: The validation dataloader.

    Returns:
    - The validation loss.
    """

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
        "log_every_n_steps": 10,
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
    assert isinstance(model, pl.LightningModule)

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
            model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt
        )
    elif args.pretrained_dir is not None:
        print("Load pre-trained weights from", args.pretrained_dir)
        if args.model_type == "Supervised":
            final_dict = clf_update_weights(args.pretrained_dir, model)
        else:
            final_dict = update_weights(args.pretrained_dir, model)
        # update initial state dict with weights from pretraining and fill the rest with initial weights
        model.load_state_dict(final_dict)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    else:
        print("No loading of pre-trained weights, start training from scratch")
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Retrieve the best validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss


def apply_mapping_from_file(adata, file_path):
    """
    Apply mapping from a file to assign integer labels to perturbations in adata.

    Args:
        adata (AnnData): Annotated data object.
        file_path (str): Path to the file containing the mapping.

    Returns:
        list: List of integer labels corresponding to the perturbations in adata.
    """
    mapping_df = pd.read_csv(file_path)
    mapping = dict(zip(mapping_df["perturbation"], mapping_df["int_label"]))
    return [
        mapping.get(perturbation, -1)
        for perturbation in adata.obs["perturbation"].values
    ]


def setup_dataloader(args, adata_train, adata_val, adata_test):
    """
    Setup the dataloader for the training script.

    Args:
        args: The command line arguments.
        adata_train: The training AnnData object.
        adata_val: The validation AnnData object.
        adata_test: The test AnnData object.

    Returns:
        train_dataloader: The DataLoader for the training dataset.
        val_dataloader: The DataLoader for the validation dataset.
    """
    # Apply mapping
    if args.split == "Default":
        train_perturbations_int = apply_mapping_from_file(
            adata_train, os.path.join(args.file_path, "pert_name_mapping.csv")
        )
        val_perturbations_int = apply_mapping_from_file(
            adata_val, os.path.join(args.file_path, "pert_name_mapping.csv")
        )
    elif args.split == "MedSplit":
        train_perturbations_int = apply_mapping_from_file(
            adata_train, os.path.join(args.file_path, "med_split_pert_name_mapping.csv")
        )
        val_perturbations_int = apply_mapping_from_file(
            adata_val, os.path.join(args.file_path, "med_split_pert_name_mapping.csv")
        )
    elif args.split == "HardSplit":
        train_perturbations_int = apply_mapping_from_file(
            adata_train,
            os.path.join(args.file_path, "hard_split_pert_name_mapping.csv"),
        )
        val_perturbations_int = apply_mapping_from_file(
            adata_val, os.path.join(args.file_path, "hard_split_pert_name_mapping.csv")
        )
    else:
        raise ValueError(
            "Unknown split. Choose Default, MedSplit or HardSplit. You chose "
            + args.split
        )
    train_dataset = AdataDataset(
        genes=torch.tensor(adata_train.X.todense()),
        perturbations=train_perturbations_int,
    )
    val_dataset = AdataDataset(
        genes=torch.tensor(adata_val.X.todense()), perturbations=val_perturbations_int
    )
    # Get dataloader
    num_cpus_available = 28  # os.cpu_count()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_cpus_available,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_cpus_available,
    )

    return train_dataloader, val_dataloader


def get_checkpoint(args):
    """
    Get the checkpoint path based on the provided arguments.

    Args:
        args (Namespace): The command line arguments.

    Returns:
        str: The checkpoint path.
    """
    # Setup checkpoint path
    dataset_name = "SciPlex2020"
    if args.pretrained_dir is None:
        model_name = "supervised"
    else:
        model_name = args.pretrained_dir.split("/")[-5]

    model_name = model_name + "_" + dataset_name + "_" + args.version

    CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, model_name)
    print("Checkpoint path is " + CHECKPOINT_PATH)
    return CHECKPOINT_PATH


if __name__ == "__main__":
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    args = parse_arguments()
    print("Root directory is " + root)

    # Setup data
    # Get Adata
    if args.split == "Default":
        print("Using default split")
        adata = sc.read_h5ad(
            os.path.join(args.adata_dir, "Srivatsan_2020_sciplex3_train_val_hvg.h5ad")
        )
        adata_test = sc.read_h5ad(
            os.path.join(args.adata_dir, "Srivatsan_2020_sciplex3_test_hvg.h5ad")
        )
    elif args.split == "MedSplit":
        print("Using medium split")
        adata = sc.read_h5ad(
            os.path.join(
                args.adata_dir, "Med_Split_Srivatsan_2020_sciplex3_train_val_hvg.h5ad"
            )
        )
        adata_test = sc.read_h5ad(
            os.path.join(
                args.adata_dir, "Med_Split_Srivatsan_2020_sciplex3_test_hvg.h5ad"
            )
        )
    elif args.split == "HardSplit":
        print("Using hard split")
        adata = sc.read_h5ad(
            os.path.join(
                args.adata_dir, "Hard_Split_Srivatsan_2020_sciplex3_train_val_hvg.h5ad"
            )
        )
        adata_test = sc.read_h5ad(
            os.path.join(
                args.adata_dir, "Hard_Split_Srivatsan_2020_sciplex3_test_hvg.h5ad"
            )
        )
    # Splitting done in the data notebook, together with holdout test set
    adata_train = adata[adata.obs["split"] == "train"]
    adata_val = adata[adata.obs["split"] == "val"]

    train_dataloader, val_dataloader = setup_dataloader(
        args, adata_train, adata_val, adata_test
    )
    # Setup model

    gene_dim = adata_train.X.shape[1]
    train_set_size = adata_train.shape[0]
    val_set_size = adata_val.shape[0]

    if args.model_type == "Supervised":
        # Get number of perturbations
        perturbations = adata_train.obs["perturbation"].values
        type_dim = len(np.unique(perturbations))
        model = PertClassifier(
            gene_dim=gene_dim,
            type_dim=type_dim,
            units=[512, 512, 256, 256, 64],
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )
    elif args.model_type == "Unsupervised":
        model = MLPAutoEncoder(
            gene_dim=gene_dim,
            units_encoder=[512, 512, 256, 256, 64],
            units_decoder=[256, 256, 512, 512],
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hvg=False,
            pert=False,  # Already set to HVGs in adata
            num_hvgs=1000,
        )
    else:
        raise ValueError(
            "Unknown model type. Choose Supervised or Unsupervised. You chose "
            + args.model_type
        )

    # Setup checkpoint path
    CHECKPOINT_PATH = get_checkpoint(args)

    # Train model
    val_loss = train(args, CHECKPOINT_PATH, model, train_dataloader, val_dataloader)
    print("Validation loss is " + str(val_loss))
