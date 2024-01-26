import numpy as np
import pandas as pd
import faiss
import os
import lightning.pytorch as pl
import torch
from self_supervision.estimator.cellnet import EstimatorAutoEncoder
from self_supervision.tester.classifier.utils import correct_labels, update_clf_report
from self_supervision.tester.classifier.model_utils import LightningWrapper
from self_supervision.tester.classifier.kNN import perform_knn


def evaluate_knn_from_file(
    reference_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    test_labels: np.ndarray,
    cell_type_hierarchy: np.ndarray,
    clf_report: pd.DataFrame,
    RESULT_PATH: str,
    res: faiss.StandardGpuResources,
    setting: str,
    k: int = 5,
    index_str: str = "PCA",
):
    """
    Perform k-NN classification using pre-computed train and test embeddings stored in .npy files.

    Parameters:
    - reference_embeddings (np.ndarray): The reference set embeddings.
    - test_embeddings (np.ndarray): The test set embeddings.
    - reference_labels (np.ndarray): The labels for the reference set embeddings.
    - test_labels (np.ndarray): The true labels for the test set.
    - cell_type_hierarchy, clf_report, RESULT_PATH, setting: Existing parameters for analysis and reporting.
    - res: FAISS GPU resources.
    - k (int): The number of neighbors for k-NN. Default is 5.

    Returns:
    - y_pred (np.ndarray): The predicted labels for the test set.
    """
    # Dimensionality of the embedding space
    d = reference_embeddings.shape[1]

    # Build the index with the reference embeddings
    index_flat = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(reference_embeddings.astype(np.float32))

    # Perform k-NN to find neighbors in the reference set for each test embedding
    _, I = gpu_index_flat.search(
        test_embeddings.astype(np.float32), k + 1
    )  # Including self-match

    # Generate predictions based on the closest neighbors in the reference set
    # Exclude the first column since it may represent the test point itself if present in the reference set
    y_pred = reference_labels[I[:, 1]]  # excluding self

    y_pred_corr = correct_labels(test_labels, y_pred, cell_type_hierarchy)

    update_clf_report(
        y_pred_corr=y_pred_corr,
        y_true=test_labels,
        model_dir=index_str,
        clf_report=clf_report,
        RESULT_PATH=RESULT_PATH,
        setting=setting,
    )

    return y_pred


def evaluate_random_model(
    estim: pl.LightningModule,
    y_true: np.ndarray,
    clf_report: pd.DataFrame,
    RESULT_PATH: str,
    cell_type_hierarchy: np.ndarray,
    reference_labels: np.ndarray,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    setting: str,
    batch_size: int,
) -> None:
    """
    Evaluate a randomly initialized model.

    Args:
        estim (pl.LightningModule): The model to evaluate.
        y_true (np.ndarray): The true labels for the data.
        clf_report (pd.DataFrame): The classification report to update.
        RESULT_PATH (str): The path to save the results.
        cell_type_hierarchy (np.ndarray): The hierarchy of cell types.
        reference_labels (np.ndarray): The labels for the reference set.
        reference_dataloader (torch.utils.data.DataLoader): The dataloader for the reference set.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test set.
        setting (str): The setting to use for evaluation.

    Returns:
        None
    """
    print("Evaluating randomly initialized model")
    # init model
    estim.init_model(
        model_type="mlp_clf",
        model_kwargs={
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "dropout": 0.1,
            "lr_scheduler_kwargs": {"step_size": 2, "gamma": 0.9, "verbose": True},
            "units": [512, 512, 256, 256, 64],
        },
    )

    # Wrap the original model with LightningWrapper
    wrapped_model = LightningWrapper(estim.model)

    # Assign the wrapped model to estim for prediction
    estim = EstimatorAutoEncoder(data_path=estim.data_path, hvg=estim.hvg)
    estim.init_datamodule(batch_size=8192)
    estim.model = wrapped_model
    estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

    print("Estim model: ", estim.model)

    if setting == "hlca":
        estim.model.base_model.supervised_subset = 148
    elif setting == "pbmc":
        estim.model.base_model.supervised_subset = 41
    elif setting == "tabula_sapiens":
        estim.model.base_model.supervised_subset = 87
    else:
        estim.model.base_model.supervised_subset = None

    estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

    # get paths

    reference_embeddings_path = os.path.join(
        RESULT_PATH,
        "classification",
        "embeddings",
        "val_emb_" + "Random_" + str(setting) + ".npy",
    )

    test_embeddings_path = os.path.join(
        RESULT_PATH,
        "classification",
        "embeddings",
        "test_emb_" + "Random_" + str(setting) + ".npy",
    )

    if os.path.exists(reference_embeddings_path):
        reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
        print("Loaded reference embeddings from disk at ", reference_embeddings_path)
    else:
        reference_embeddings = estim.predict_embedding(val_dataloader)
        np.save(reference_embeddings_path, reference_embeddings)
        print("Saved reference embeddings to disk at ", reference_embeddings_path)

    if os.path.exists(test_embeddings_path):
        test_embeddings = np.load(test_embeddings_path, mmap_mode="r")
        print("Loaded test embeddings from disk at ", test_embeddings_path)
    else:
        test_embeddings = estim.predict_embedding(test_dataloader)
        np.save(test_embeddings_path, test_embeddings)
        print("Saved test embeddings to disk at ", test_embeddings_path)

    assert test_embeddings.shape[0] == y_true.shape[0], (
        "Test embeddings and labels do not match in shape. Your shape of the test embeddings is: "
        + str(test_embeddings.shape)
        + " and the shape of the test labels is: "
        + str(y_true.shape)
    )
    assert reference_embeddings.shape[0] == reference_labels.shape[0], (
        "Reference embeddings and labels do not match in shape. Your shape of the reference embeddings is: "
        + str(reference_embeddings.shape)
        + " and the shape of the reference labels is: "
        + str(reference_labels.shape)
    )

    y_pred = perform_knn(
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        test_embeddings=test_embeddings,
    )

    # y_pred = perform_knn(clf_embeddings, y_true, res)
    y_pred_corr = correct_labels(y_true, y_pred, cell_type_hierarchy)

    update_clf_report(
        y_pred_corr=y_pred_corr,
        y_true=y_true,
        model_dir="Random",
        clf_report=clf_report,
        RESULT_PATH=RESULT_PATH,
        setting=setting,
    )
