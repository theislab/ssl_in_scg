import argparse
import os
from ast import literal_eval
from pathlib import Path
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from self_supervision.data.checkpoint_utils import (
    load_last_checkpoint,
    checkpoint_exists,
)
from self_supervision.estimator.cellnet import EstimatorAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_dir", default=None, type=str)
    parser.add_argument(
        "--supervised_subset",
        default=None,
        type=str,
        choices=[None, "HLCA", "PBMC", "Tabula_Sapiens", "PBMCs_Integration"],
        help="Dataset name",
    )
    parser.add_argument("--batch_size", default=16384, type=int)
    parser.add_argument(
        "--hidden_units", default=[512, 512, 256, 256, 64], type=literal_eval
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--version", default="", type=str)
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
        "--stochastic", action="store_true", help="Whether to use a random seed"
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
    parser.add_argument(
    "--log_freq",
    default=10,
    type=int,
    help="logging frequency",
    )
    parser.add_argument(
        "--checkpoint_interval", default=1, type=int, help="Checkpoint interval"
    )
    parser.add_argument(
    "--max_steps",
    default=123794,
    type=int,
    help="number of max epochs before stopping training",
    )

    return parser.parse_args()


def update_weights(pretrained_dir, estim):
    """
    Update the weights of the model_dict based on possible matches in the pretrained dict
    Correct for some diverging layer-names
    :param pretrained_dir: Directory of the pretrained-weights
    :param estim: Estimator object
    :return: Dictionary of final model weights, include pretrained and randomly initialized weights
    """
    # load and init dicts
    try:
        pretrained_dict = torch.load(pretrained_dir)["state_dict"]
    except KeyError:  # for manual saving, it's directly the state dict
        pretrained_dict = torch.load(pretrained_dir)
    model_dict = estim.model.state_dict()
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


if __name__ == "__main__":
    # parse args
    args = parse_args()
    print(args)

    # reproducibility
    if not args.stochastic:
        torch.manual_seed(0)

    if args.pretrained_dir:
        if "reconstruction" in args.pretrained_dir:
            pre_model = (
                args.pretrained_dir.split("/")[-6]
                + "_"
                + args.pretrained_dir.split("/")[-5]
            )
        elif "bt_Gaussian" in args.pretrained_dir:
            pre_model = (
                args.pretrained_dir.split("/")[-3]
                + "_"
                + args.pretrained_dir.split("/")[-2]
            )
        else:
            pre_model = args.pretrained_dir.split("/")[-5]
    else:
        pre_model = None

    # CHECKPOINT HANDLING
    if args.hvg:
        use_hvg = "HVG_" + str(args.num_hvgs) + "_"
    else:
        use_hvg = ""
    if pre_model:
        subfolder = "SSL_" + use_hvg + pre_model + args.version
    else:
        subfolder = "No_SSL_" + use_hvg + args.version

    subfolder = subfolder + f"_{args.supervised_subset}"
    if args.supervised_subset == "HLCA":
        dataset_ids = [148]
    elif args.supervised_subset == "Tabula_Sapiens":
        dataset_ids = [87]
    elif args.supervised_subset == "PBMC":
        dataset_ids = [41]
    elif args.supervised_subset == "PBMCs_Integration":
        dataset_ids = [243, 179, 225]
    else:
        dataset_ids = None

    CHECKPOINT_PATH = os.path.join(
        args.model_path, "trained_models", "final_models", "classification", subfolder
    )
    print("Will save model to " + CHECKPOINT_PATH)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    # get estimator
    estim = EstimatorAutoEncoder(
        data_path=args.data_path, hvg=args.hvg, num_hvgs=args.num_hvgs
    )

    # set up datamodule
    estim.init_datamodule(batch_size=args.batch_size)

    estim.init_trainer(
        trainer_kwargs={
            "max_steps": args.max_steps,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "default_root_dir": CHECKPOINT_PATH,
            "accelerator": "gpu",
            "devices": 1,
            "num_sanity_val_steps": 0,
            "check_val_every_n_epoch": 1,
            "logger": [WandbLogger(save_dir=CHECKPOINT_PATH)],
            "log_every_n_steps": args.log_freq,
            "detect_anomaly": False,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "enable_checkpointing": True,
            "callbacks": [
                TQDMProgressBar(refresh_rate=50),
                LearningRateMonitor(logging_interval="step"),
                # Save the model with the best training loss
                ModelCheckpoint(
                    filename="best_checkpoint_train",
                    monitor="train_loss",
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
    )

    # init model
    estim.init_model(
        model_type="mlp_clf",
        model_kwargs={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "dropout": args.dropout,
            "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
            "units": args.hidden_units,
            "supervised_subset": dataset_ids,
            "num_hvgs": args.num_hvgs,
        },
    )

    # load model from SSL pretraining if it's the first time this fine-tuning is done - else load finetuning ckpt
    if not (not args.pretrained_dir or checkpoint_exists(CHECKPOINT_PATH)):
        print("Load pre-trained weights from", args.pretrained_dir)
        final_dict = update_weights(args.pretrained_dir, estim)
        # update initial state dict with weights from pretraining and fill the rest with initial weights
        estim.model.load_state_dict(final_dict)
        print("Model type: ", type(estim.model))
        estim.train()
    else:
        # check if there are .ckpt files in the checkpoint directory
        if checkpoint_exists(CHECKPOINT_PATH):
            print(
                "No loading of pre-trained weights, continue training from",
                CHECKPOINT_PATH,
            )
            # get best checkpoint from previous training with PyTorch Lightning
            # ckpt = load_best_checkpoint(CHECKPOINT_PATH)
            ckpt = load_last_checkpoint(CHECKPOINT_PATH)
            print("Load checkpoint from", ckpt)
            estim.train(ckpt_path=ckpt)
        else:
            print("No loading of pre-trained weights, start training from scratch")
            estim.train()
