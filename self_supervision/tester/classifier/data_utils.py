import torch
import os
import numpy as np
import dask.dataframe as dd
from typing import List
import lightning.pytorch as pl


def load_true_labels(
    DATA_PATH: str,
    REFERENCE_PATH: str,
    test_dataloader: torch.utils.data.DataLoader,
    setting: str,
) -> np.array:
    """
    Load true labels from the given data path.

    Args:
        DATA_PATH (str): The path to the data directory.
        REFERENCE_PATH (str): The path to the reference directory.
        test_dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
        setting (str): The setting for loading the labels.

    Returns:
        tuple: A tuple containing the train labels, validation labels, and test labels.
    """
    # Load true labels
    train_labels = (
        dd.read_parquet(os.path.join(REFERENCE_PATH, "train"), columns=["cell_type"])
        .compute()
        .to_numpy()
    )

    val_labels = (
        dd.read_parquet(os.path.join(REFERENCE_PATH, "val"), columns=["cell_type"])
        .compute()
        .to_numpy()
    )

    if (
        setting == "CellNet"
        or setting == "hlca"
        or setting == "pbmc"
        or setting == "tabula_sapiens"
    ):
        test_labels_reference = dd.read_parquet(
            os.path.join(DATA_PATH, "test"), columns=["dataset_id"]
        )
        val_labels_reference = dd.read_parquet(
            os.path.join(DATA_PATH, "val"), columns=["dataset_id"]
        )
        test_labels = dd.read_parquet(
            os.path.join(DATA_PATH, "test"), columns=["cell_type"]
        )
        val_labels = dd.read_parquet(
            os.path.join(DATA_PATH, "val"), columns=["cell_type"]
        )
        if setting == "hlca":
            test_labels = (
                test_labels[test_labels_reference["dataset_id"] == 148]
                .compute()
                .to_numpy()
            )
            val_labels = (
                val_labels[val_labels_reference["dataset_id"] == 148]
                .compute()
                .to_numpy()
            )
            print(
                "Am in hlca setting, have {} test and {} val labels left".format(
                    test_labels.shape[0], val_labels.shape[0]
                )
            )
        elif setting == "pbmc":
            test_labels = (
                test_labels[test_labels_reference["dataset_id"] == 41]
                .compute()
                .to_numpy()
            )
            val_labels = (
                val_labels[val_labels_reference["dataset_id"] == 41]
                .compute()
                .to_numpy()
            )
            print(
                "Am in pbmc setting, have {} test and {} val labels left".format(
                    test_labels.shape[0], val_labels.shape[0]
                )
            )
        elif setting == "tabula_sapiens":
            test_labels = (
                test_labels[test_labels_reference["dataset_id"] == 87]
                .compute()
                .to_numpy()
            )
            val_labels = (
                val_labels[val_labels_reference["dataset_id"] == 87]
                .compute()
                .to_numpy()
            )
            print(
                "Am in tabula_sapiens setting, have {} test and {} val labels left".format(
                    test_labels.shape[0], val_labels.shape[0]
                )
            )
        else:
            test_labels = test_labels.compute().to_numpy()
            val_labels = val_labels.compute().to_numpy()
    else:
        if isinstance(test_dataloader.dataset.tensor_y, List):
            test_labels = np.array(test_dataloader.dataset.tensor_y)
        else:
            test_labels = test_dataloader.dataset.tensor_y.numpy()
    return train_labels, val_labels, test_labels


def get_embeddings(
    estim: pl.LightningModule,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    RESULT_PATH: str,
    setting: str,
    index_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes or loads embeddings for training, validation, and test data.

    Args:
        estim (pl.LightningModule): The model to use for computing embeddings.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
        val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        RESULT_PATH (str): The path to the directory where the embeddings will be saved.
        reference (str): The reference data to use for computing embeddings. Must be one of 'train', 'val', or 'combined'.
        setting (str): Unused.
        index_str (str): A string to append to the saved file names.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the reference embeddings and test embeddings.
    """

    if setting == "CellNet":
        test_embeddings_path = os.path.join(
            RESULT_PATH,
            "classification",
            "embeddings",
            "test_emb_" + index_str + ".npy",
        )
        val_embeddings_path = os.path.join(
            RESULT_PATH, "classification", "embeddings", "val_emb_" + index_str + ".npy"
        )

    else:
        test_embeddings_path = os.path.join(
            RESULT_PATH,
            "classification",
            "embeddings",
            "test_emb_" + index_str + "_" + setting + ".npy",
        )
        val_embeddings_path = os.path.join(
            RESULT_PATH,
            "classification",
            "embeddings",
            "val_emb_" + index_str + "_" + setting + ".npy",
        )

    if os.path.exists(val_embeddings_path):
        reference_embeddings = np.load(val_embeddings_path, mmap_mode="r")
        print("Loaded val embeddings from disk at ", val_embeddings_path)
    else:
        # double check that supervised subset is on
        if setting == "hlca":
            estim.model.supervised_subset = 148
        elif setting == "pbmc":
            estim.model.supervised_subset = 41
        elif setting == "tabula_sapiens":
            estim.model.supervised_subset = 87
        else:
            estim.model.supervised_subset = None
        reference_embeddings = estim.predict_embedding(val_dataloader)
        np.save(val_embeddings_path, reference_embeddings)
        print("Saved val embeddings to disk at ", val_embeddings_path)

    if os.path.exists(test_embeddings_path):
        test_embeddings = np.load(test_embeddings_path, mmap_mode="r")
        print("Loaded test embeddings from disk at ", test_embeddings_path)
    else:
        test_embeddings = estim.predict_embedding(test_dataloader)
        np.save(test_embeddings_path, test_embeddings)
        print("Saved test embeddings to disk at ", test_embeddings_path)

    return reference_embeddings, test_embeddings


def load_and_process_labels(data_path, reference_path, test_dataloader, setting):
    """
    Load and process labels for training, validation, and testing.

    Args:
        data_path (str): Path to the data.
        reference_path (str): Path to the reference data.
        test_dataloader (DataLoader): DataLoader for the test data.
        setting (str): Setting for label processing.

    Returns:
        tuple: A tuple containing the processed training labels, validation labels, and test labels.
    """
    train_labels, val_labels, test_labels = load_true_labels(
        data_path, reference_path, test_dataloader, setting
    )
    return np.squeeze(train_labels), np.squeeze(val_labels), np.squeeze(test_labels)


def prepare_dataloader(estim, train_dataloader, val_dataloader, test_dataloader):
    """
    Prepares the dataloader for training, validation, and testing.

    Args:
        estim: The estimator object.
        train_dataloader: The dataloader for training data. If None, it will be fetched from estim.datamodule.train_dataloader().
        val_dataloader: The dataloader for validation data. If None, it will be fetched from estim.datamodule.val_dataloader().
        test_dataloader: The dataloader for testing data. If None, it will be fetched from estim.datamodule.test_dataloader().

    Returns:
        train_dataloader: The prepared dataloader for training data.
        val_dataloader: The prepared dataloader for validation data.
        test_dataloader: The prepared dataloader for testing data.
    """
    if train_dataloader is None:
        train_dataloader = estim.datamodule.train_dataloader()
    if val_dataloader is None:
        val_dataloader = estim.datamodule.val_dataloader()
    if test_dataloader is None:
        test_dataloader = estim.datamodule.test_dataloader()

    return train_dataloader, val_dataloader, test_dataloader


def prepare_data_path(DATA_PATH, setting):
    """
    Prepare the data path based on the specified setting.

    Args:
        DATA_PATH (str): The base data path.
        setting (str): The setting to determine the specific data path.

    Returns:
        tuple: A tuple containing the reference path and the data path.

    Raises:
        ValueError: If the specified setting requires a specific dataset that is not found.

    """
    REFERENCE_PATH = os.path.join(DATA_PATH, "merlin_cxg_2023_05_15_sf-log1p")
    if (
        setting == "CellNet"
        or setting == "hlca"
        or setting == "pbmc"
        or setting == "tabula_sapiens"
    ):
        DATA_PATH = os.path.join(DATA_PATH, "merlin_cxg_2023_05_15_sf-log1p")
    elif setting == "OOD_HiT":
        DATA_PATH = os.path.join(DATA_PATH, "tail_of_hippocampus")
        if not os.path.exists(DATA_PATH):
            raise ValueError("OOD_HiT dataset not found. Please download it first.")
    elif setting == "OOD_nn":
        DATA_PATH = os.path.join(DATA_PATH, "non_neuronal")
        if not os.path.exists(DATA_PATH):
            raise ValueError("OOD_nn dataset not found. Please download it first.")
    elif setting == "OOD_Circ_Imm":
        DATA_PATH = os.path.join(DATA_PATH, "circ_imm")
        if not os.path.exists(DATA_PATH):
            raise ValueError(
                "OOD_Circ_Imm dataset not found. Please download it first."
            )
    elif setting == "OOD_Cort_Dev":
        DATA_PATH = os.path.join(DATA_PATH, "cort_dev")
        if not os.path.exists(DATA_PATH):
            raise ValueError(
                "OOD_Cort_Dev dataset not found. Please download it first."
            )
    elif setting == "OOD_Great_Apes":
        DATA_PATH = os.path.join(DATA_PATH, "great_apes")
        if not os.path.exists(DATA_PATH):
            raise ValueError(
                "OOD_Great_Apes dataset not found. Please download it first."
            )
    return REFERENCE_PATH, DATA_PATH


def initialize_subset(setting):
    """
    Initializes the subset based on the given setting.

    Args:
        setting (str): The setting to initialize the subset for.

    Returns:
        int: The initialized subset value.

    Raises:
        None

    """
    if setting == "hlca":
        return 148
    elif setting == "pbmc":
        return 41
    elif setting == "tabula_sapiens":
        return 87
    else:
        return None


def get_pca_embeddings(
    DATA_PATH: str,
    setting: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PCA embeddings based on the specified setting.

    Args:
        DATA_PATH (str): The path to the data.
        setting (str): The setting for loading the PCA embeddings.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the reference embeddings and test embeddings.
    """

    # Load PCA embeddings
    if setting == "CellNet":
        val_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_val_64.npy",
        )
        test_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_test_64.npy",
        )
    elif setting == "OOD_HiT":
        val_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_val_64.npy",
        )
        test_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "tail_of_hippocampus",
            "x_tail_of_hippocampus_pca_64.npy",
        )
    elif setting == "OOD_nn":
        val_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_val_64.npy",
        )
        test_pca_file = os.path.join(
            os.path.dirname(DATA_PATH), "non_neuronal", "x_non_neuronal_pca_64.npy"
        )
    elif setting == "OOD_Circ_Imm":
        val_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_val_64.npy",
        )
        test_pca_file = os.path.join(
            os.path.dirname(DATA_PATH), "circ_imm", "x_circ_imm_pca_64.npy"
        )
    elif setting == "OOD_Cort_Dev":
        val_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_val_64.npy",
        )
        test_pca_file = os.path.join(
            os.path.dirname(DATA_PATH), "cort_dev", "x_cort_dev_pca_64.npy"
        )
    elif setting == "OOD_Great_Apes":
        val_pca_file = os.path.join(
            os.path.dirname(DATA_PATH),
            "merlin_cxg_2023_05_15_sf-log1p",
            "x_pca_val_64.npy",
        )
        test_pca_file = os.path.join(
            os.path.dirname(DATA_PATH), "great_apes", "x_great_apes_pca_64.npy"
        )
    else:
        raise ValueError(
            "Invalid setting. Must be one of CellNet, OOD_HiT, OOD_nn. You provided: ",
            setting,
        )

    reference_embeddings = np.load(val_pca_file, mmap_mode="r")
    print("Loaded val embeddings from disk at ", val_pca_file)

    test_embeddings = np.load(test_pca_file, mmap_mode="r")
    print("Loaded test embeddings from disk at ", test_pca_file)

    return reference_embeddings, test_embeddings


def load_from_file(eval_file):
    """
    Load data from files.

    Args:
        eval_file (tuple): A tuple containing file paths for reference embeddings,
                           reference labels, test embeddings, and test labels.

    Returns:
        tuple: A tuple containing reference embeddings, reference labels,
               test embeddings, and test labels.
    Raises:
        ValueError: If any of the file paths are not a string or a numpy array.
    """

    reference_emb_path = eval_file[0]
    if isinstance(reference_emb_path, str):
        reference_embeddings = np.load(reference_emb_path, mmap_mode="r")
    elif isinstance(reference_emb_path, np.ndarray):
        reference_embeddings = reference_emb_path
    else:
        raise ValueError("reference_emb_path must be either a string or a numpy array.")

    reference_labels_path = eval_file[1]
    if isinstance(reference_labels_path, str):
        reference_labels = np.load(reference_labels_path, mmap_mode="r")
    elif isinstance(reference_labels_path, np.ndarray):
        reference_labels = reference_labels_path
    else:
        raise ValueError(
            "reference_labels_path must be either a string or a numpy array."
        )

    test_emb_path = eval_file[2]
    if isinstance(test_emb_path, str):
        test_embeddings = np.load(test_emb_path, mmap_mode="r")
    elif isinstance(test_emb_path, np.ndarray):
        test_embeddings = test_emb_path
    else:
        raise ValueError("test_emb_path must be either a string or a numpy array.")

    test_labels_path = eval_file[3]
    if isinstance(test_labels_path, str):
        test_labels = np.load(test_labels_path, mmap_mode="r")
    elif isinstance(test_labels_path, np.ndarray):
        test_labels = test_labels_path
    else:
        raise ValueError("test_labels_path must be either a string or a numpy array.")
    
    # Check that the shapes match
    assert reference_embeddings.shape[0] == reference_labels.shape[0], (
        "Reference embeddings and labels do not match in shape. Your shape of the reference embeddings is: "
        + str(reference_embeddings.shape)
        + " and the shape of the reference labels is: "
        + str(reference_labels.shape)
    )
    assert test_embeddings.shape[0] == test_labels.shape[0], (
        "Test embeddings and labels do not match in shape. Your shape of the test embeddings is: "
        + str(test_embeddings.shape)
        + " and the shape of the test labels is: "
        + str(test_labels.shape)
    )

    return reference_embeddings, reference_labels, test_embeddings, test_labels
