from typing import Tuple
from sklearn.metrics import classification_report
import lightning.pytorch as pl
import numpy as np
import os
import pandas as pd
import dask.dataframe as dd
import torch
from self_supervision.models.lightning_modules.cellnet_autoencoder import MLPClassifier
from self_supervision.estimator.cellnet import EstimatorAutoEncoder
from self_supervision.tester.classifier.test import correct_labels
from self_supervision.paths import DATA_DIR, RESULTS_FOLDER, TRAINING_FOLDER


def generate_data_for_visualization(
    estim, model_dirs: list[str], DATA_PATH: str, supervised_subset: int = None
):
    f1_scores = {}
    cell_counts = {}

    for model_name, model_dir in zip(["Model A", "Model B"], model_dirs):
        # Load model checkpoint
        estim.model = MLPClassifier.load_from_checkpoint(
            model_dir,
            **estim.get_fixed_clf_params(),
            supervised_subset=supervised_subset,
            units=[512, 512, 256, 256, 64],
        )
        estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

        # Predict on test data
        y_pred, y_true = predict_custom_dataloader(
            estim=estim,
            dataloader=estim.datamodule.test_dataloader(),
            supervised_subset=supervised_subset,
        )
        assert y_pred.shape[0] == y_true.shape[0]

        # Correct the labels
        cell_type_hierarchy = np.load(
            os.path.join(DATA_PATH, "cell_type_hierarchy/child_matrix.npy")
        )
        y_pred_corr = correct_labels(y_true, y_pred, cell_type_hierarchy)

        # Get the classification report
        clf_report = pd.DataFrame(
            classification_report(y_true, y_pred_corr, output_dict=True)
        ).T

        print("Overall clf report: ", clf_report, "\n")

        # Subset the report to only include the cell types that are present in the test set (y_true)
        clf_report = clf_report.loc[y_true.unique()]

        print("Subsetted clf report: ", clf_report, "\n")

        # F1 scores for each cell type
        per_class_f1 = clf_report["f1-score"].to_dict()
        f1_scores[model_name] = per_class_f1

        # Cell counts for each cell type
        cell_counts_dict = dict(zip(*np.unique(y_true, return_counts=True)))
        cell_counts[model_name] = cell_counts_dict

    return f1_scores, cell_counts


def old_generate_data_for_visualization(
    estim, model_dirs: list[str], DATA_PATH: str, supervised_subset: int = None
):
    """
    Generates data for visualization for the given model directories.
    Args:
        - estim: EstimatorAutoEncoder instance
        - model_dirs: A list containing paths to model directories
        - DATA_PATH: Path to the data
    Returns:
        - f1_scores: A dictionary containing F1-scores for each cell type for both models.
        - cell_counts: A dictionary containing counts of unique cells per cell type.
    """
    f1_scores = {}
    cell_counts = {}

    for model_name, model_dir in zip(["Model A", "Model B"], model_dirs):
        # Load model checkpoint
        estim.model = MLPClassifier.load_from_checkpoint(
            model_dir,
            **estim.get_fixed_clf_params(),
            supervised_subset=supervised_subset,
            units=[512, 512, 256, 256, 64],
        )
        estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

        # Predict on test data
        y_pred, y_true = predict_custom_dataloader(
            estim=estim,
            dataloader=estim.datamodule.test_dataloader(),
            supervised_subset=supervised_subset,
        )
        assert y_pred.shape[0] == y_true.shape[0]

        hypothetical_report_dict_no_corr = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        print(
            "Report dict for model " + model_name + ":\n",
            hypothetical_report_dict_no_corr,
            "\n",
        )

        # Correct the labels
        cell_type_hierarchy = np.load(
            os.path.join(DATA_PATH, "cell_type_hierarchy/child_matrix.npy")
        )
        y_pred_corr = correct_labels(y_true, y_pred, cell_type_hierarchy)

        # Get classification report
        unique_y_true = np.unique(y_true)
        report_dict = classification_report(
            y_true, y_pred_corr, labels=unique_y_true, output_dict=True, zero_division=0
        )

        print("Report dict for model " + model_name + ":\n", report_dict, "\n")

        # F1 scores for each cell type
        per_class_f1 = {
            str(cell_type): report_dict.get(str(cell_type), {"f1-score": 0.0})[
                "f1-score"
            ]
            for cell_type in unique_y_true
        }
        f1_scores[model_name] = per_class_f1

        # Cell counts for each cell type
        cell_counts_dict = dict(zip(*np.unique(y_true, return_counts=True)))
        cell_counts[model_name] = cell_counts_dict

    return f1_scores, cell_counts


def get_histogram(
    supervised_subset: int,
    supervised_dir: str,
    ssl_dir: str,
):
    """
    Generate a histogram for classification results.

    Args:
        estim (EstimatorAutoEncoder): The estimator object.
        supervised_subset (int, optional): The subset ID for supervised training. Defaults to None.
        ssl_dir (str): The directory of the self-supervised model.
    Returns:
        pd.DataFrame: The generated histogram data.
    """
    STORE_DIR = os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p")
    # init estim class
    estim = EstimatorAutoEncoder(STORE_DIR, hvg=False)

    # init datamodule
    estim.init_datamodule(batch_size=8192)

    supervised_dir = TRAINING_FOLDER + supervised_dir
    ssl_dir = TRAINING_FOLDER + ssl_dir

    # Load labels
    test_labels_reference = dd.read_parquet(
        os.path.join(STORE_DIR, "test"), columns=["dataset_id"]
    )
    test_labels = dd.read_parquet(
        os.path.join(STORE_DIR, "test"), columns=["cell_type"]
    )
    test_labels = (
        test_labels[test_labels_reference["dataset_id"] == supervised_subset]
        .compute()
        .to_numpy()
    )

    # Eval models
    f1_scores, cell_counts = generate_data_for_visualization(
        estim=estim,
        model_dirs=[supervised_dir, ssl_dir],
        DATA_PATH=STORE_DIR,
        supervised_subset=supervised_subset,
    )

    # Store as dataframe
    # Convert dictionaries into DataFrame
    print("Cell types: ", len(list(cell_counts["Model A"].keys())))
    print("Cell types: ", list(cell_counts["Model A"].keys()))
    print("F1 scores: ", len(list(f1_scores["Model A"].values())))
    print("F1 score per cell type: ", list(f1_scores["Model A"].values()))
    print("SSL F1 scores: ", len(list(f1_scores["Model B"].values())))
    print("SSL F1 score per cell type: ", list(f1_scores["Model B"].values()))
    print("Cell counts: ", len(list(cell_counts["Model A"].values())))

    df = pd.DataFrame(
        {
            "Cell_Type": list(cell_counts["Model A"].keys()),
            "Supervised_F1": list(f1_scores["Model A"].values()),
            "Self-Supervised_F1": list(f1_scores["Model B"].values()),
            "Cell_Count": list(cell_counts["Model A"].values()),
        }
    )

    # Add F1_Difference column
    df["F1_Difference"] = df["Self-Supervised_F1"] - df["Supervised_F1"]

    # Store dataframe
    file_path = os.path.join(
        RESULTS_FOLDER, "classification", "histogram_" + str(supervised_subset) + ".csv"
    )
    df.to_csv(file_path, index=False)
    print("Saved histogram to " + file_path)
    return df


def predict_custom_dataloader(
    estim, dataloader=None, supervised_subset: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    estim._check_is_initialized()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_idx = 0
    for batch in dataloader:
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]
        # Send every tensor to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Apply dataset_id filtering only if supervised_subset is set
        if supervised_subset is not None:
            # print('batch: ', batch)
            mask = batch["dataset_id"] == supervised_subset
            if not any(mask):
                return None, None  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            filtered_batch = {key: value[mask] for key, value in batch.items()}
        else:
            filtered_batch = batch

        # Compute reconstructed transcriptomes for both models
        preds, true = estim.model.predict_step(
            batch=filtered_batch, batch_idx=batch_idx
        )
        # Concatenate the latent space and reconstructed transcriptomes
        if batch_idx == 0:
            all_preds = preds
            all_true = true
        else:
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_true = torch.cat((all_true, true), dim=0)

        batch_idx += 1

    # Convert to numpy
    all_preds = all_preds.cpu().detach().numpy()
    all_true = all_true.cpu().detach().numpy()

    return all_preds, all_true
