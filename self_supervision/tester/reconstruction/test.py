import numpy as np
import pandas as pd
import os
import lightning.pytorch as pl
import torch
from self_supervision.models.lightning_modules.cellnet_autoencoder import (
    MLPAutoEncoder,
    MLPNegBin,
)
from self_supervision.trainer.reconstruction.train import update_weights
from self_supervision.estimator.cellnet import EstimatorAutoEncoder
from self_supervision.data.datamodules import get_large_ood_dataloader
from self_supervision.paths import RESULTS_FOLDER, DATA_DIR, OOD_FOLDER
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection
import pickle
import dask.dataframe as dd
import h5py


def eval_rec(
    estim,
    model_dirs: list[str],
    supervised_subset: int = None,
    eval_random: bool = False,
) -> None:
    """
    Evaluate all models in model_dirs
    :param estim: EstimatorAutoEncoder
    :param model_dirs: list of model directories
    :return: None
    """
    # Create a new DataFrame to store the flattened data
    if supervised_subset is None:
        rec_report_path = os.path.join(RESULTS_FOLDER, "reconstruction", "rec_report.csv")
    else:
        rec_report_path = os.path.join(
            RESULTS_FOLDER,
            "reconstruction",
            "rec_report_" + str(supervised_subset) + ".csv",
        )
    if not os.path.exists(rec_report_path):
        rec_report = pd.DataFrame(
            columns=["Explained Variance Uniform", "Explained Variance Weighted", "MSE"]
        )
        rec_report.index.name = "experiment"
        rec_report.to_csv(rec_report_path)
    else:
        rec_report = pd.read_csv(rec_report_path, index_col=0)

    for model_dir in model_dirs:
        subdir = model_dir.split("/")[-5]
        # skip if subdir is already in rec_report
        if subdir in rec_report.index:
            print("Skipping model: ", subdir)
            continue

        # Init estimator
        estim = EstimatorAutoEncoder(
            data_path=os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p"),
            hvg=False,
        )

        estim.init_datamodule(batch_size=1024)

        # init model
        hidden_units = [512, 512, 256, 256, 64]
        estim.init_model(
            model_type="mlp_negbin" if "NegBin" in model_dir else "mlp_ae",
            model_kwargs={
                "learning_rate": 1e-3,
                "weight_decay": 0.1,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR,
                "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
                "units_encoder": hidden_units,
                "units_decoder": hidden_units[::-1][1:],
            },
        )

        print("Evaluating model: ", subdir)

        # Load model checkpoint
        if "NegBin" in model_dir:
            try:
                estim.model = MLPNegBin.load_from_checkpoint(
                    model_dir,
                    **estim.get_fixed_autoencoder_params(),
                    units_encoder=[512, 512, 256, 256, 64],
                    units_decoder=[256, 256, 512, 512],
                    supervised_subset=supervised_subset,
                )
            except Exception as e:
                print("Could not load model: ", subdir)
                print(e)
                try:
                    # Load model checkpoint
                    final_dict = update_weights(
                        model_dir,
                        estim,
                        model_type="NegBin" if "NegBin" in model_dir else "MLP",
                    )
                    # update initial state dict with weights from pretraining and fill the rest with initial weights
                    estim.model.load_state_dict(final_dict)
                    estim.model.supervised_subset = supervised_subset
                except Exception as e:
                    print("Could not load model: ", subdir)
                    print(e)
                    continue

        else:
            try:
                estim.model = MLPAutoEncoder.load_from_checkpoint(
                    model_dir,
                    **estim.get_fixed_autoencoder_params(),
                    units_encoder=[512, 512, 256, 256, 64],
                    units_decoder=[256, 256, 512, 512],
                    supervised_subset=supervised_subset,
                )
            except Exception as e:
                print("Could not load model: ", subdir)
                print(e)
                try:
                    # Load model checkpoint
                    final_dict = update_weights(model_dir, estim)
                    # update initial state dict with weights from pretraining and fill the rest with initial weights
                    estim.model.load_state_dict(final_dict)
                    estim.model.supervised_subset = supervised_subset
                except Exception as e:
                    print("Could not load model: ", subdir)
                    print(e)
                    continue

        estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

        metrics = estim.test()
        print("Metrics for model: ", subdir, " :\n", metrics)
        explained_var_uni_i = metrics[0]["test_explained_var_uniform"]
        explained_var_w_i = metrics[0]["test_explained_var_weighted"]
        mse_i = metrics[0]["test_mse"]

        # add it to the report
        rec_report.loc[subdir] = [explained_var_uni_i, explained_var_w_i, mse_i]

        # save report
        rec_report.to_csv(rec_report_path)

        # Avoid the error super(type, obj): obj must be an instance or subtype of type
        del estim.model

    # Evaluate randomly initialized model
    if eval_random:
        del estim
        estim = EstimatorAutoEncoder(
            data_path=os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p"),
            hvg=False,
        )

        estim.init_datamodule(batch_size=1024)

        # init model
        hidden_units = [512, 512, 256, 256, 64]
        estim.init_model(
            model_type="mlp_negbin" if "NegBin" in model_dir else "mlp_ae",
            model_kwargs={
                "learning_rate": 1e-3,
                "weight_decay": 0.1,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR,
                "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
                "units_encoder": hidden_units,
                "units_decoder": hidden_units[::-1][1:],
            },
        )

        estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

        metrics = estim.test()
        print("Metrics for model: ", "Random", " :\n", metrics)
        explained_var_uni_i = metrics[0]["test_explained_var_uniform"]
        explained_var_w_i = metrics[0]["test_explained_var_weighted"]
        mse_i = metrics[0]["test_mse"]

        # add it to the report
        rec_report.loc["Random"] = [explained_var_uni_i, explained_var_w_i, mse_i]

        # save report
        rec_report.to_csv(rec_report_path)
    return None


def test_random_model(
    estim,
    supervised_subset: int = None,
) -> None:
    """
    Test a randomly initialized model
    :param estim: EstimatorAutoEncoder
    :return: None
    """
    # Create a new DataFrame to store the flattened data
    rec_report_path = os.path.join(
        RESULTS_FOLDER, "reconstruction", "rec_report_comp_ssl.csv"
    )
    if not os.path.exists(rec_report_path):
        rec_report = pd.DataFrame(
            columns=["Explained Variance Uniform", "Explained Variance Weighted", "MSE"]
        )
        rec_report.index.name = "experiment"
        rec_report.to_csv(rec_report_path)
    else:
        rec_report = pd.read_csv(rec_report_path, index_col=0)

    # Evaluate randomly initialized model
    del estim.model
    estim.init_model(
        model_type="mlp_ae",
        model_kwargs={
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
            "units_encoder": [512, 512, 256, 256, 64],
            "units_decoder": [256, 256, 512, 512],
            "supervised_subset": supervised_subset,
        },
    )
    estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

    metrics = estim.test()
    explained_var_uni_i = metrics[0]["test_explained_var_uniform"]
    explained_var_w_i = metrics[0]["test_explained_var_weighted"]
    mse_i = metrics[0]["test_mse"]

    # add it to the report
    rec_report.loc["Random"] = [explained_var_uni_i, explained_var_w_i, mse_i]

    # save report
    rec_report.to_csv(rec_report_path)

    return None


def test_pretrained_model(
    estim,
    model_dir: str,
) -> None:
    """
    Test a pretrained model that was not finetuned for the reconstruction task
    :param estim: EstimatorAutoEncoder
    :param model_dir: directory of the model
    :return: None
    """
    # Create a new DataFrame to store the flattened data
    rec_report_path = os.path.join(
        RESULTS_FOLDER, "reconstruction", "rec_report_comp_ssl.csv"
    )
    if not os.path.exists(rec_report_path):
        rec_report = pd.DataFrame(
            columns=["Explained Variance Uniform", "Explained Variance Weighted", "MSE"]
        )
        rec_report.index.name = "experiment"
        rec_report.to_csv(rec_report_path)
    else:
        rec_report = pd.read_csv(rec_report_path, index_col=0)

    # Load model checkpoint
    final_dict = update_weights(model_dir, estim)
    # update initial state dict with weights from pretraining and fill the rest with initial weights
    estim.model.load_state_dict(final_dict)

    estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

    metrics = estim.test()
    explained_var_uni_i = metrics[0]["test_explained_var_uniform"]
    explained_var_w_i = metrics[0]["test_explained_var_weighted"]
    mse_i = metrics[0]["test_mse"]

    # add it to the report
    rec_report.loc["Only Pretrained"] = [explained_var_uni_i, explained_var_w_i, mse_i]

    # add index str
    rec_report.index = ["Only Pretrained"]

    # save report
    rec_report.to_csv(rec_report_path)

    return None


def test_models_on_ood(
    estim,
    ood_set: str,
    model_dirs: list[str],
) -> None:
    """
    Test all models in model_dirs on the OOD dataset
    Use get_large_ood_dataloader(x_file_path, y_file_path, batch_size) to get the dataloader
    :param estim: EstimatorAutoEncoder
    :param model_dirs: list of model directories
    :return: None
    """
    # Create a new DataFrame to store the flattened data
    rec_report_path = os.path.join(
        RESULTS_FOLDER, "reconstruction", "rec_report_comp_" + ood_set + ".csv"
    )
    if not os.path.exists(rec_report_path):
        rec_report = pd.DataFrame(
            columns=["Explained Variance Uniform", "Explained Variance Weighted", "MSE"]
        )
        rec_report.index.name = "experiment"
        rec_report.to_csv(rec_report_path)
    else:
        rec_report = pd.read_csv(rec_report_path, index_col=0)

    for model_dir in model_dirs:
        subdir = model_dir.split("/")[-5]
        # skip if subdir is already in rec_report
        if subdir in rec_report.index:
            print("Skipping model: ", subdir)
            continue

        print("Evaluating model: ", subdir)

        # Load model checkpoint
        if "NegBin" in model_dir:
            try:
                estim.model = MLPNegBin.load_from_checkpoint(
                    model_dir,
                    **estim.get_fixed_autoencoder_params(),
                    units_encoder=[512, 512, 256, 256, 64],
                    units_decoder=[256, 256, 512, 512],
                )
            except Exception as e:
                print("Could not load model: ", subdir)
                print(e)
                try:
                    # Load model checkpoint
                    final_dict = update_weights(
                        model_dir,
                        estim,
                        model_type="NegBin" if "NegBin" in model_dir else "MLP",
                    )
                    # update initial state dict with weights from pretraining and fill the rest with initial weights
                    estim.model.load_state_dict(final_dict)
                except Exception as e:
                    print("Could not load model: ", subdir)
                    print(e)
                    continue

        else:
            try:
                estim.model = MLPAutoEncoder.load_from_checkpoint(
                    model_dir,
                    **estim.get_fixed_autoencoder_params(),
                    units_encoder=[512, 512, 256, 256, 64],
                    units_decoder=[256, 256, 512, 512],
                )
            except Exception as e:
                print("Could not load model: ", subdir)
                print(e)
                try:
                    # Load model checkpoint
                    final_dict = update_weights(model_dir, estim)
                    # update initial state dict with weights from pretraining and fill the rest with initial weights
                    estim.model.load_state_dict(final_dict)
                except Exception as e:
                    print("Could not load model: ", subdir)
                    print(e)
                    continue

        estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

        # Load OOD data
        x_file_path = os.path.join(OOD_FOLDER, ood_set, "processed_tensor_x.hdf5")
        y_file_path = os.path.join(OOD_FOLDER, ood_set, "processed_tensor_y.hdf5")
        # Converting .pt to .hdf5
        if not os.path.exists(x_file_path) and os.path.exists(
            x_file_path.replace(".hdf5", ".pt")
        ):
            print("Converting .pt to .hdf5")
            x = torch.load(x_file_path.replace(".hdf5", ".pt")).numpy()
            y = torch.load(y_file_path.replace(".hdf5", ".pt")).numpy()
            with h5py.File(x_file_path, "w") as f:
                f.create_dataset(
                    "processed_tensor_x", data=x
                )  # Changed from 'x' to 'processed_tensor_x'
            with h5py.File(y_file_path, "w") as f:
                f.create_dataset(
                    "processed_tensor_y", data=y
                )  # Changed from 'y' to 'processed_tensor_y'

        ood_dataloader = get_large_ood_dataloader(
            x_file_path, y_file_path, batch_size=1024
        )

        metrics = estim.test(dataloader=ood_dataloader)
        explained_var_uni_i = metrics[0]["test_explained_var_uniform"]
        explained_var_w_i = metrics[0]["test_explained_var_weighted"]
        mse_i = metrics[0]["test_mse"]

        # add it to the report
        rec_report.loc[subdir] = [explained_var_uni_i, explained_var_w_i, mse_i]

        # save report
        rec_report.to_csv(rec_report_path)

    # Evaluate randomly initialized model
    del estim.model
    estim.init_model(
        model_type="mlp_ae",
        model_kwargs={
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
            "units_encoder": [512, 512, 256, 256, 64],
            "units_decoder": [256, 256, 512, 512],
        },
    )
    estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

    # Load OOD data
    x_file_path = os.path.join(OOD_FOLDER, ood_set, "processed_tensor_x.hdf5")
    y_file_path = os.path.join(OOD_FOLDER, ood_set, "processed_tensor_y.hdf5")
    # Converting .pt to .hdf5
    if not os.path.exists(x_file_path) and os.path.exists(
        x_file_path.replace(".hdf5", ".pt")
    ):
        print("Converting .pt to .hdf5")
        x = torch.load(x_file_path.replace(".hdf5", ".pt")).numpy()
        y = torch.load(y_file_path.replace(".hdf5", ".pt")).numpy()
        with h5py.File(x_file_path, "w") as f:
            f.create_dataset(
                "processed_tensor_x", data=x
            )  # Changed from 'x' to 'processed_tensor_x'
        with h5py.File(y_file_path, "w") as f:
            f.create_dataset(
                "processed_tensor_y", data=y
            )  # Changed from 'y' to 'processed_tensor_y'

    ood_dataloader = get_large_ood_dataloader(x_file_path, y_file_path, batch_size=1024)

    metrics = estim.test(dataloader=ood_dataloader)
    explained_var_uni_i = metrics[0]["test_explained_var_uniform"]
    explained_var_w_i = metrics[0]["test_explained_var_weighted"]
    mse_i = metrics[0]["test_mse"]

    # add it to the report
    rec_report.loc["Random"] = [explained_var_uni_i, explained_var_w_i, mse_i]

    # save report
    rec_report.to_csv(rec_report_path)


def get_count_matrix(ddf):
    x = (
        ddf["X"]
        .map_partitions(
            lambda xx: pd.DataFrame(np.vstack(xx.tolist())),
            meta={col: "f4" for col in range(19357)},
        )
        .to_dask_array(lengths=[1024] * ddf.npartitions)
    )

    return x


def pca(
) -> None:
    """
    Evaluate the stored reconstruction from inverse PCA on PCA space
    :param RESULT_PATH: directory in which both the PCA reconstruction and the results table are stored
    :param DATA_DIR: directory in which the data is stored
    :return:
    """
    # Load PCA reconstruction
    pca_dir = os.path.join(
        RESULTS_FOLDER, "reconstruction", "reconstructed_data", "test_pca.npy"
    )
    if not os.path.exists(pca_dir):
        raise FileNotFoundError("PCA reconstruction not found")

    # Load HVG indices
    if not os.path.exists(
        os.path.join(DATA_DIR, "hvg_indices.pickle")
    ):
        raise FileNotFoundError("hvg_indices.pickle does not exist")
    hvg_indices = pickle.load(
        open(os.path.join(DATA_DIR, "hvg_indices.pickle"), "rb")
    )

    # Create a new DataFrame to store the flattened data
    rec_report_path = os.path.join(RESULTS_FOLDER, "reconstruction", "rec_report.csv")
    if not os.path.exists(rec_report_path):
        rec_report = pd.DataFrame(
            columns=["Explained Variance Uniform", "Explained Variance Weighted", "MSE"]
        )
        rec_report.index.name = "experiment"
        rec_report.to_csv(rec_report_path)
    else:
        rec_report = pd.read_csv(rec_report_path, index_col=0)

    # If PCA has already been evaluated, skip
    if "PCA" in rec_report.index:
        print("Skipping PCA")
        return None

    # Load PCA reconstruction
    pca_rec = np.load(pca_dir)

    # Load True data
    print("Load data...")
    # Load the data
    ddf_split = dd.read_parquet(os.path.join(DATA_DIR, "test"))
    x_split = get_count_matrix(ddf_split)
    del ddf_split  # delete dask dataframe to free up memory
    print("Input data: ", x_split.shape)

    # Subset the data rows to the hvg_indices
    x_split_sub = x_split[:, hvg_indices]
    del x_split  # delete x_split to free up memory
    print("Subset data: ", x_split_sub.shape)

    # Cast numpy arrays to torch tensors
    print("Type of x split: ", type(x_split_sub))
    x_split_sub = torch.tensor(x_split_sub.compute())
    pca_rec = torch.tensor(pca_rec)

    # Evaluate the reconstruction
    metrics = MetricCollection(
        {
            "explained_var_weighted": ExplainedVariance(
                multioutput="variance_weighted"
            ),
            "explained_var_uniform": ExplainedVariance(multioutput="uniform_average"),
            "mse": MeanSquaredError(),
        }
    )

    metrics.update(x_split_sub, pca_rec)
    print("Explained Variance Uniform: ", metrics["explained_var_uniform"].compute())
    print("Explained Variance Weighted: ", metrics["explained_var_weighted"].compute())
    print("MSE: ", metrics["mse"].compute())

    # add it to the report
    rec_report.loc["PCA"] = [
        metrics["explained_var_uniform"].compute(),
        metrics["explained_var_weighted"].compute(),
        metrics["mse"].compute(),
    ]

    # save report
    rec_report.to_csv(rec_report_path)
