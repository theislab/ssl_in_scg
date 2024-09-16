import argparse
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
import os
from pathlib import Path
from self_supervision.data.checkpoint_utils import (
    load_best_checkpoint,
    checkpoint_exists,
)
from self_supervision.estimator.cellnet import EstimatorAutoEncoder


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

    # iterate over lal keys in model_dict
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
            # if (model_type == 'NegBin' and pre_key == 'decoder.0.12.weight') or (model_type == 'NegBin' and pre_key == 'decoder.0.12.bias'):
            #     print('Manual exception for decoder.0.12.weight. Initializing randomly.')
            #     final_dict.update({final_key: model_dict[final_key]})
            #     continue
            # check if key is in pretrained_dict
            if pre_key == final_key:
                print("weight transfer: Direct match at", pre_key)
                # uptade final dict with pretrained_dict
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


def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="MLP",
        choices=["MLP", "VAE", "NegBin", "NegBinVAE"],
        help="Model to use",
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default=None,
        choices=["simple_vae", "scvi_vae"],
        help="VAE type",
    )
    parser.add_argument("--decoder", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--loss_func", type=str, default="MSE"
    )  # To Do: think about different loss functions - CB, MSE
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--pretrained_dir", type=str, default=None)
    parser.add_argument("--hidden_units", type=list, default=[512, 512, 256, 256, 64])
    parser.add_argument(
        "--supervised_subset",
        default=None,
        type=str,
        # choices=[None, 148, 87, 41],
        choices=[None, 'HLCA', 'Tabula_Sapiens', 'PBMC']
    )
    # Checkpointing
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--checkpoint_interval", type=int, default=1)
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
        "--pert", action="store_true", help="Whether to use a random seed"
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
    "--max_steps",
    default=123794,
    type=int,
    help="number of max epochs before stopping training",
    )
    parser.add_argument(
    "--log_freq",
    default=10,
    type=int,
    help="logging frequency",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # GET GPU AND ARGS
    # if torch.cuda.is_available():
    #     print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()
    print(args)

    # reproducibility
    if not args.stochastic:
        torch.manual_seed(0)

    # CHECKPOINT HANDLING
    if args.hvg:
        use_hvg = "HVG_"
    elif args.pert:
        use_hvg = "PERT_"
        args.hvg = True  # For the Estimator
        num_hvgs = 1000  # For the Estimator
    else:
        use_hvg = ""
        num_hvgs = None

    if args.pretrained_dir:
        if "classification" in args.pretrained_dir:
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

    if pre_model:
        subfolder = "SSL_CN_" + pre_model + args.version
    else:
        subfolder = "No_SSL_CN_" + args.version

    subfolder = subfolder + args.model + "_"

    if args.supervised_subset == "HLCA":
        subfolder = subfolder + "_HLCA"
        supervised_subset = 148
    elif args.supervised_subset == "Tabula_Sapiens":
        subfolder = subfolder + "_Tabula_Sapiens"
        supervised_subset = 87
    elif args.supervised_subset == "PBMC":
        subfolder = subfolder + "_PBMC"
        supervised_subset = 41
    else:
        supervised_subset=None

    CHECKPOINT_PATH = os.path.join(
        args.model_path,
        "trained_models",
        "final_models",
        "reconstruction",
        "CN_" + use_hvg + subfolder,
    )
    print("Will save model to", CHECKPOINT_PATH)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    # get estimator
    estim = EstimatorAutoEncoder(
        data_path=args.data_path, hvg=args.hvg, num_hvgs=num_hvgs
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

    # init model
    if args.model == "MLP":
        model_type = "mlp_ae"
    elif args.model == "VAE":
        model_type = "mlp_vae"
    elif args.model == "NegBin":
        model_type = "mlp_negbin"
    elif args.model == "NegBinVAE":
        model_type = "mlp_negbin_vae"
    else:
        raise ValueError(
            "Model not recognized. You can choose between MLP, VAE, NegBin, NegBinVAE. You chose",
            args.model,
        )

    estim.init_model(
        model_type=model_type,
        model_kwargs={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
            "units_encoder": args.hidden_units,
            "units_decoder": args.hidden_units[::-1][1:] if args.decoder else [],
            "supervised_subset": supervised_subset,
        },
    )

    print(ModelSummary(estim.model))

    # load model from SSL pretraining if it's the first time this fine-tuning is done - else load finetuning ckpt
    # a valid checkpoint file is at CHECKPOINT_PATH + 'some subfolder' + '/best_val_loss.ckpt
    # check if there is a valid checkpoint file in the folder or subfolders

    if not (not args.pretrained_dir or checkpoint_exists(CHECKPOINT_PATH)):
        print("Load pre-trained weights from", args.pretrained_dir)
        final_dict = update_weights(args.pretrained_dir, estim)
        # update initial state dict with weights from pretraining and fill the rest with initial weights
        estim.model.load_state_dict(final_dict)
        estim.train()
    else:
        if checkpoint_exists(CHECKPOINT_PATH):
            print(
                "No loading of pre-trained weights, continue training from",
                CHECKPOINT_PATH,
            )
            # get the best checkpoint from previous training with PyTorch Lightning
            ckpt = load_best_checkpoint(CHECKPOINT_PATH)
            print("Load checkpoint from", ckpt)
            estim.train(ckpt_path=ckpt)
        else:
            print("No loading of pre-trained weights, start training from scratch")
            estim.train()
