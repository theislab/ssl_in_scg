import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from numba import njit
from sklearn.metrics import classification_report
from  self_supervision.paths import DATA_DIR, RESULTS_FOLDER


@njit
def correct_labels(y_true: np.ndarray, y_pred: np.ndarray, child_matrix: np.ndarray):
    """
    Update predictions.
    If prediction is actually a child node of the true label -> update prediction to true value.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        child_matrix (np.ndarray): Matrix representing the child-parent relationship between labels.

    Returns:
        np.ndarray: Array of updated predictions.

    Example:
        >>> y_true = np.array([0, 1, 2, 3])
        >>> y_pred = np.array([1, 1, 2, 4])
        >>> child_matrix = np.array([[0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0]])
        >>> correct_labels(y_true, y_pred, child_matrix)
        array([0, 1, 2, 4])
    """
    updated_predictions = y_pred.copy()
    # precalculate child nodes
    child_nodes = {
        i: np.where(child_matrix[i, :])[0] for i in range(child_matrix.shape[0])
    }

    for i, (pred, true_label) in enumerate(zip(y_pred, y_true)):
        if pred in child_nodes[true_label]:
            updated_predictions[i] = true_label
        else:
            updated_predictions[i] = pred

    return updated_predictions


def prepare_clf_report(
    setting: str = "CellNet",
    reference: str = "val",
) -> pd.DataFrame:
    """
    Prepare the classification report for a given setting and reference.

    Args:
        setting (str): The setting for the classification report. Default is 'CellNet'.
        reference (str): The reference for the classification report. Default is 'val'.

    Returns:
        pd.DataFrame: The classification report as a pandas DataFrame.

    """
    clf_report_path = os.path.join(
        RESULTS_FOLDER, "classification", reference + "_clf_report_" + setting + "_knn.csv"
    )
    if not os.path.exists(clf_report_path):
        clf_report = pd.DataFrame(
            columns=[
                "precision: accuracy",
                "precision: macro avg",
                "precision: weighted avg",
                "recall: accuracy",
                "recall: macro avg",
                "recall: weighted avg",
                "f1-score: accuracy",
                "f1-score: macro avg",
                "f1-score: weighted avg",
                "support: accuracy",
                "support: macro avg",
                "support: weighted avg",
            ]
        )
        clf_report.index.name = "experiment"
        clf_report.to_csv(clf_report_path)
        print("Save new clf_report to ", clf_report_path)
        return clf_report
    else:
        return pd.read_csv(clf_report_path, index_col=0)


def prepare_per_class_clf_report(
    supervised_subset: int = None,
) -> pd.DataFrame:
    """
    Prepare the per-class classification report.

    Args:
        supervised_subset (int): Subset ID for supervised training. Default is None.

    Returns:
        pd.DataFrame: The per-class classification report.

    """
    STORE_DIR = os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p")
    # Get all labels for the given setting
    # Load true labels
    if supervised_subset is None:
        all_labels = (
            dd.read_parquet(os.path.join(STORE_DIR, "labels"), columns=["cell_type"])
            .compute()
            .to_numpy()
        )
    else:
        test_labels_reference = dd.read_parquet(
            os.path.join(STORE_DIR, "test"), columns=["dataset_id"]
        )
        all_labels = dd.read_parquet(
            os.path.join(STORE_DIR, "test"), columns=["cell_type"]
        )
        all_labels = (
            all_labels[test_labels_reference["dataset_id"] == supervised_subset]
            .compute()
            .to_numpy()
        )

    clf_report_path = os.path.join(
        RESULTS_FOLDER,
        "classification",
        "per_class_" + str(supervised_subset) + "_per_class_clf_report_knn.csv",
    )

    if not os.path.exists(clf_report_path):
        clf_report = pd.DataFrame(
            columns=[
                "Cell Type",
                "Supervised F1",
                "Self-Supervised F1",
                "Cell Count",
                "F1 Difference",
            ]
        )
        clf_report.index.name = "cell_type"
        clf_report.to_csv(clf_report_path)
        print("Save new clf_report to ", clf_report_path)
        return clf_report

    else:
        return pd.read_csv(clf_report_path, index_col=0)


def get_index_str(model_dir: str) -> str:
    """
    Returns the index string based on the given model directory.

    Parameters:
    model_dir (str): The directory of the model.

    Returns:
    str: The index string.

    Raises:
    ValueError: If the model directory does not contain a valid model.
    """
    index_str = model_dir.split("/")[-5] if model_dir != "Random" else "Random"

    if "pretext_model" in model_dir:
        print("Get index for pretrained")
        index_str = f"{index_str}_Only Pretrained"
    elif "final_model" in model_dir and "No_SSL" in model_dir:
        print("Get index for no ssl")
        index_str = f"{index_str}_No SSL"
    elif model_dir == "Random":
        print("Get index for Random")
        index_str = "Random"
    elif "final_model" in model_dir and "No_SSL" not in model_dir:
        print("Get index for ssl")
        index_str = f"{index_str}_SSL"
    else:
        raise ValueError("Model directory does not contain a valid model.")

    return index_str


def update_clf_report(
    y_pred_corr: np.array,
    y_true: np.array,
    model_dir: str,
    clf_report: pd.DataFrame,
    RESULT_PATH: str,
    setting: str,
) -> pd.DataFrame:
    """
    Update the classification report with the predicted and true labels for a given model.

    Args:
        y_pred_corr (np.array): Array of predicted labels.
        y_true (np.array): Array of true labels.
        model_dir (str): Directory of the model.
        clf_report (pd.DataFrame): Existing classification report.
        RESULT_PATH (str): Path to the result directory.
        setting (str): Setting for the classification report.

    Returns:
        pd.DataFrame: Updated classification report.

    """
    clf_report_i = pd.DataFrame(
        classification_report(y_true, y_pred_corr, output_dict=True)
    ).T
    clf_report_i_overall = clf_report_i.iloc[-3:].copy()
    flatten_data = clf_report_i_overall.T.values.flatten()
    flatten_df = pd.DataFrame(flatten_data.reshape(1, -1), columns=clf_report.columns)
    if model_dir == "PCA":
        index_str = "PCA"
    elif model_dir == "PCA_Test":
        index_str = "PCA_Test"
    elif model_dir == "GeneFormer":
        index_str = "GeneFormer"
    elif model_dir == "Random":
        index_str = "Random"
    else:
        index_str = get_index_str(model_dir)
    flatten_df.index = [index_str]
    clf_report = pd.concat([clf_report, flatten_df], axis=0)
    clf_report = clf_report.sort_index()
    current_report = prepare_clf_report(RESULT_PATH=RESULT_PATH, setting=setting)
    # Append the flattened DataFrame to clf_report but with pd.concat
    final_report = pd.concat([current_report, clf_report], axis=0)

    # sort by index name and save
    final_report = final_report.sort_index()
    final_report.to_csv(
        os.path.join(
            RESULT_PATH, "classification", "val_clf_report_" + setting + "_knn.csv"
        )
    )
    print(
        "Saved ",
        index_str,
        " to ",
        os.path.join(
            RESULT_PATH, "classification", "val_clf_report_" + setting + "_knn.csv"
        ),
    )
    return clf_report


def update_per_class_clf_report(
    y_pred_corr: np.array,
    y_true: np.array,
    is_supervised: bool = False,
) -> None:
    """
    Update the classification report DataFrame based on the corrected predicted labels.

    Args:
        y_pred_corr (np.array): Array of corrected predicted labels.
        y_true (np.array): Array of true labels.
        model_dir (str): Directory path where the model is saved.
        clf_report_path (str): Path to save the updated classification report.
        supervised_subset (int): Number of samples in the supervised subset.
        is_supervised (bool, optional): Flag indicating whether the model is supervised or self-supervised.
            Defaults to False.

    Returns:
        clf_report (pd.DataFrame): Updated classification report DataFrame.
    """
    # Create and transpose the classification report
    clf_report_i = pd.DataFrame(
        classification_report(y_true, y_pred_corr, output_dict=True)
    ).T

    # Extract the F1 scores for each class (excluding overall metrics)
    f1_scores = clf_report_i["f1-score"].drop(["accuracy", "macro avg", "weighted avg"])

    # Define the column name for F1 scores based on the model type
    f1_column = "Supervised F1" if is_supervised else "Self-Supervised F1"

    # Create a DataFrame with F1 scores
    clf_report_i_per_class_df = pd.DataFrame(
        {f1_column: f1_scores.values}, index=f1_scores.index
    )

    clf_report = clf_report_i_per_class_df
    return clf_report
