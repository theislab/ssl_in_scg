import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import faiss
from self_supervision.data.datamodules import AdataDataset
from sklearn.metrics import classification_report
import lightning.pytorch as pl
from self_supervision.models.lightning_modules.cellnet_autoencoder import MLPAutoEncoder
from self_supervision.trainer.perturbations.train import update_weights
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import gc


# kNN Evaluation


def prepare_clf_report(
    RESULT_PATH: str = "/lustre/groups/ml01/workspace/till.richter/ssl_results",
    setting: str = "CellNet",
) -> pd.DataFrame:
    """
    Initialize or load the classification report DataFrame.
    """
    clf_report_path = os.path.join(
        RESULT_PATH, "perturbations", "pert_report_" + setting + "_knn.csv"
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


def apply_mapping_from_file(adata, file_path):
    mapping_df = pd.read_csv(file_path)
    mapping = dict(zip(mapping_df["perturbation"], mapping_df["int_label"]))
    return [
        mapping.get(perturbation, -1)
        for perturbation in adata.obs["perturbation"].values
    ]


def setup_dataloader(
    adata_dir: str = "/lustre/groups/ml01/workspace/till.richter/",
    file_path: str = "/home/icb/till.richter/git/self_supervision/self_supervision/data/perturbations",
    batch_size: int = 1024,
    setting: str = "SciPlex2020",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Setup the dataloader for the training script.
    """

    if "SciPlex" in setting:
        test_adata_dir = os.path.join(
            adata_dir, "Srivatsan_2020_sciplex3_test_hvg.h5ad"
        )
        train_val_adata_dir = os.path.join(
            adata_dir, "Srivatsan_2020_sciplex3_train_val_hvg.h5ad"
        )

        # Val loader (for kNN reference)
        adata = sc.read_h5ad(train_val_adata_dir)
        adata_val = adata[adata.obs["split"] == "val"]
        del adata

        # Test loader
        adata = sc.read_h5ad(test_adata_dir)

        test_perturbations_int = apply_mapping_from_file(
            adata, os.path.join(file_path, "pert_name_mapping.csv")
        )

        test_dataset = AdataDataset(
            genes=torch.tensor(adata.X.todense()), perturbations=test_perturbations_int
        )

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Get all perturbations from adata and subset adata_val to only contain those
        test_perturbations = adata.obs["perturbation"].unique()
        adata_val = adata_val[adata_val.obs["perturbation"].isin(test_perturbations)]

        val_perturbations_int = apply_mapping_from_file(
            adata_val, os.path.join(file_path, "pert_name_mapping.csv")
        )

        val_dataset = AdataDataset(
            genes=torch.tensor(adata_val.X.todense()),
            perturbations=val_perturbations_int,
        )

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    else:
        raise ValueError("Unknown dataset.")

    return val_dataloader, test_dataloader


def load_true_labels(
    adata_val,
    adata_test,
    file_path: str = "/home/icb/till.richter/git/self_supervision/self_supervision/data/perturbations",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load true labels from the given data path.
    """
    # Create val adata subset that only contains the perturbations that are also in the test set
    test_perturbations = adata_test.obs["perturbation"].unique()
    adata_val = adata_val[adata_val.obs["perturbation"].isin(test_perturbations)]
    # Load true labels
    val_perturbations_int = apply_mapping_from_file(
        adata_val, os.path.join(file_path, "pert_name_mapping.csv")
    )
    test_perturbations_int = apply_mapping_from_file(
        adata_test, os.path.join(file_path, "pert_name_mapping.csv")
    )
    return val_perturbations_int, test_perturbations_int


def get_index_str(model_dir: str) -> str:
    """
    Helper function to get the index string for the classification report.
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


def update_pert_report(
    y_pred_corr: np.array,
    y_true: np.array,
    model_dir: str,
    clf_report: pd.DataFrame,
    RESULT_PATH: str,
    setting: str,
) -> pd.DataFrame:
    """
    Update the perturbation report DataFrame based on the corrected predicted labels.
    :type model_dir: str
    :return Updated clf_report DataFrame
    """
    clf_report_i = pd.DataFrame(
        classification_report(y_true, y_pred_corr, output_dict=True)
    ).T
    clf_report_i_overall = clf_report_i.iloc[-3:].copy()
    flatten_data = clf_report_i_overall.T.values.flatten()
    flatten_df = pd.DataFrame(flatten_data.reshape(1, -1), columns=clf_report.columns)
    if model_dir == "PCA":
        index_str = "PCA"
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
            RESULT_PATH, "perturbations", "pert_report_" + setting + "_knn.csv"
        )
    )
    print(
        "Saved ",
        index_str,
        " to ",
        os.path.join(
            RESULT_PATH, "perturbation", "pert_report_" + setting + "_knn.csv"
        ),
    )
    return clf_report


def load_model(
    model_dir: str,
    gene_dim: int,
    train_set_size: int,
    val_set_size: int,
    batch_size: int,
) -> pl.LightningModule:
    if "final_model" in model_dir:
        model = MLPAutoEncoder.load_from_checkpoint(
            model_dir,
            gene_dim=gene_dim,
            units_encoder=[512, 512, 256, 256, 64],
            units_decoder=[256, 256, 512, 512],
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=batch_size,
            hvg=False,
            pert=False,
            num_hvgs=1000,
        )
        return model

    elif "pretext_model" in model_dir:
        model = MLPAutoEncoder(
            gene_dim=gene_dim,
            units_encoder=[512, 512, 256, 256, 64],
            units_decoder=[256, 256, 512, 512],
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=batch_size,
            hvg=False,
            pert=False,
            num_hvgs=1000,
        )
        final_dict = update_weights(model_dir, model)
        model.load_state_dict(final_dict)
        return model
    else:
        raise ValueError("Model directory does not contain a valid model.")


def perform_knn(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Perform k-Nearest Neighbors classification using train embeddings to classify test embeddings.

    Args:
        reference_embeddings (numpy.ndarray): The embeddings of the reference set.
        reference_labels (numpy.ndarray): The corresponding labels for each reference embedding.
        test_embeddings (numpy.ndarray): The embeddings of the test set to classify.
        k (int): Number of nearest neighbors to use for classification.
        use_gpu (bool): Whether to use GPU resources for Faiss index.

    Returns:
        numpy.ndarray: The predicted labels for each test embedding.
    """
    print("Start kNN...")
    # Ensure that the embeddings and labels are in float32 for FAISS
    reference_embeddings = reference_embeddings.astype(np.float32)
    test_embeddings = test_embeddings.astype(np.float32)

    # Initialize the index
    d = reference_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)

    if use_gpu:
        # Initialize GPU resources and transfer the index to GPU
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add reference embeddings to the index
    index.add(reference_embeddings)

    # Perform the search for the k-nearest neighbors
    # Note: We search for `k + 1` neighbors because the closest point is expected to be the point itself
    D, I = index.search(test_embeddings, k + 1)

    # Process the results:
    # Exclude the first column of indices (self-match)
    # and use the remaining columns to vote for the predicted label
    y_pred = np.array(
        [np.argmax(np.bincount(reference_labels[I[i, 1:]])) for i in range(I.shape[0])]
    )

    return y_pred


def evaluate_random_model(
    y_true: np.ndarray,
    clf_report: pd.DataFrame,
    RESULT_PATH: str,
    reference_labels: np.ndarray,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    setting: str,
    gene_dim: int,
    train_set_size: int,
    val_set_size: int,
    batch_size: int,
) -> None:
    """
    Evaluate a randomly initialized model.

    Args:
        y_true (np.ndarray): The true labels for the data.
        clf_report (pd.DataFrame): The classification report to update.
        RESULT_PATH (str): The path to save the results.
        reference_labels (np.ndarray): The labels for the reference set.
        reference_dataloader (torch.utils.data.DataLoader): The dataloader for the reference set.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test set.
        setting (str): The setting to use for evaluation.

    Returns:
        None
    """
    print("Evaluating randomly initialized model")

    model = MLPAutoEncoder(
        gene_dim=gene_dim,
        units_encoder=[512, 512, 256, 256, 64],
        units_decoder=[256, 256, 512, 512],
        train_set_size=train_set_size,
        val_set_size=val_set_size,
        batch_size=batch_size,
        hvg=False,
        pert=False,
        num_hvgs=1000,  # this is ugly ugly code
    )

    # get paths
    val_embeddings_path = os.path.join(
        RESULT_PATH,
        "perturbations",
        "embeddings",
        "val_emb_" + "Random" + str(setting) + ".npy",
    )

    test_embeddings_path = os.path.join(
        RESULT_PATH,
        "perturbations",
        "embeddings",
        "test_emb_" + "Random" + str(setting) + ".npy",
    )

    reference_dataloader = val_dataloader
    reference_embeddings_path = val_embeddings_path

    if os.path.exists(reference_embeddings_path):
        reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
        print("Loaded reference embeddings from disk at ", reference_embeddings_path)
    else:
        # Initialize an empty list to store embeddings
        embeddings_list = []

        # Iterate over each batch in the DataLoader
        for batch in reference_dataloader:
            # Predict the embeddings for the current batch
            batch_embeddings = model.predict_embedding(batch)
            # Put batch embeddings on CPU and convert to NumPy array
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            # Append the batch embeddings to the list
            embeddings_list.append(batch_embeddings)

        # Concatenate all batch embeddings
        reference_embeddings = np.concatenate(embeddings_list, axis=0)
        np.save(reference_embeddings_path, reference_embeddings)

        print("Saved reference embeddings to disk at ", reference_embeddings_path)

    if os.path.exists(test_embeddings_path):
        test_embeddings = np.load(test_embeddings_path, mmap_mode="r")
        print("Loaded test embeddings from disk at ", test_embeddings_path)
    else:
        embeddings_list = []
        for batch in test_dataloader:
            # Predict the embeddings for the current batch
            batch_embeddings = model.predict_embedding(batch)
            # Put batch embeddings on CPU and convert to NumPy array
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            # Append the batch embeddings to the list
            embeddings_list.append(batch_embeddings)

        # Concatenate all batch embeddings
        test_embeddings = np.concatenate(embeddings_list, axis=0)
        np.save(test_embeddings_path, test_embeddings)
        print("Saved test embeddings to disk at ", test_embeddings_path)

    assert reference_embeddings.shape[0] == reference_labels.shape[0], (
        "Reference embeddings and labels do not match in shape. Your shape of the reference embeddings is: "
        + str(reference_embeddings.shape)
        + " and the shape of the reference labels is: "
        + str(reference_labels.shape)
    )
    assert test_embeddings.shape[0] == y_true.shape[0], (
        "Test embeddings and labels do not match in shape. Your shape of the test embeddings is: "
        + str(test_embeddings.shape)
        + " and the shape of the test labels is: "
        + str(y_true.shape)
    )

    y_pred = perform_knn(
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        test_embeddings=test_embeddings,
    )

    update_pert_report(
        y_pred_corr=y_pred,
        y_true=y_true,
        model_dir="Random",
        clf_report=clf_report,
        RESULT_PATH=RESULT_PATH,
        setting=setting,
    )


def get_embeddings(
    model,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    RESULT_PATH: str,
    setting: str,
    index_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes or loads embeddings for training, validation, and test data.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
        val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        RESULT_PATH (str): The path to the directory where the embeddings will be saved.
        setting (str): Unused.
        index_str (str): A string to append to the saved file names.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the reference embeddings and test embeddings.
    """
    # Compute or load embeddings
    test_embeddings_path = os.path.join(
        RESULT_PATH, "classification", "embeddings", "test_emb_" + index_str + ".npy"
    )
    val_embeddings_path = os.path.join(
        RESULT_PATH, "classification", "embeddings", "val_emb_" + index_str + ".npy"
    )

    if os.path.exists(val_embeddings_path):
        reference_embeddings = np.load(val_embeddings_path, mmap_mode="r")
        print("Loaded val embeddings from disk at ", val_embeddings_path)
    else:
        # Initialize an empty list to store embeddings
        embeddings_list = []
        for batch in val_dataloader:
            # Place all elements in the batch on the GPU
            for key in batch:
                batch[key] = batch[key].to(model.device)
            # Predict the embeddings for the current batch
            reference_embeddings = model.predict_embedding(batch)
            # Put batch embeddings on CPU and convert to NumPy array
            reference_embeddings = reference_embeddings.detach().cpu().numpy()
            # Append the batch embeddings to the list
            embeddings_list.append(reference_embeddings)

        # Concatenate all batch embeddings
        reference_embeddings = np.concatenate(embeddings_list, axis=0)
        np.save(val_embeddings_path, reference_embeddings)
        print("Saved val embeddings to disk at ", val_embeddings_path)

    if os.path.exists(test_embeddings_path):
        test_embeddings = np.load(test_embeddings_path, mmap_mode="r")
        print("Loaded test embeddings from disk at ", test_embeddings_path)
    else:
        embeddings_list = []
        for batch in test_dataloader:
            # Place all elements in the batch on the GPU
            for key in batch:
                batch[key] = batch[key].to(model.device)
            # Predict the embeddings for the current batch
            test_embeddings = model.predict_embedding(batch)
            # Put batch embeddings on CPU and convert to NumPy array
            test_embeddings = test_embeddings.detach().cpu().numpy()
            # Append the batch embeddings to the list
            embeddings_list.append(test_embeddings)

        # Concatenate all batch embeddings
        test_embeddings = np.concatenate(embeddings_list, axis=0)
        np.save(test_embeddings_path, test_embeddings)
        print("Saved test embeddings to disk at ", test_embeddings_path)

    return reference_embeddings, test_embeddings


def eval_emb_knn(
    model_dirs: list[str],
    DATA_PATH: str = "/lustre/groups/ml01/workspace/till.richter/",
    eval_random: bool = True,
    batch_size: int = 1024,
    eval_file: Optional[Tuple[str, str, str, str]] = None,
    setting: str = "SciPlex2020",
) -> None:
    # Prepare paths
    RESULT_PATH = os.path.join(DATA_PATH, "ssl_results")
    train_val_adata_dir = os.path.join(
        DATA_PATH, "Srivatsan_2020_sciplex3_train_val_hvg.h5ad"
    )
    test_adata_dir = os.path.join(DATA_PATH, "Srivatsan_2020_sciplex3_test_hvg.h5ad")

    # Load data
    adata_train_val = sc.read_h5ad(train_val_adata_dir)
    adata_val = adata_train_val[adata_train_val.obs["split"] == "val"]
    adata_train = adata_train_val[adata_train_val.obs["split"] == "train"]
    del adata_train_val
    adata_test = sc.read_h5ad(test_adata_dir)

    # Prepare dataloader:
    val_dataloader, test_dataloader = setup_dataloader()
    gene_dim = adata_train.X.shape[1]
    train_set_size = adata_train.shape[0]
    val_set_size = adata_val.shape[0]

    model = MLPAutoEncoder(
        gene_dim=gene_dim,
        units_encoder=[512, 512, 256, 256, 64],
        units_decoder=[256, 256, 512, 512],
        train_set_size=train_set_size,
        val_set_size=val_set_size,
        batch_size=batch_size,
        hvg=False,
        pert=False,
        num_hvgs=1000,  # this is ugly ugly code
    )

    # Load classification report
    clf_report = prepare_clf_report(RESULT_PATH=RESULT_PATH, setting=setting)

    # Load true labels
    val_labels, test_labels = load_true_labels(
        adata_val=adata_val, adata_test=adata_test
    )

    # Ensure labels are one-dimensional
    val_labels = np.squeeze(val_labels)
    test_labels = np.squeeze(test_labels)

    print("Loaded val labels: ", val_labels.shape, "test labels: ", test_labels.shape)

    reference_labels = val_labels

    # Evaluate models
    with ThreadPoolExecutor() as executor:
        future_to_model_dir = {}
        for model_dir in model_dirs:
            print(f"Evaluating {model_dir}")

            # Skip if already evaluated
            index_str = get_index_str(model_dir)
            if index_str in clf_report.index:
                print(f"Skipping {index_str}")
                continue

            # Load model
            model = load_model(
                model_dir=model_dir,
                gene_dim=gene_dim,
                train_set_size=train_set_size,
                val_set_size=val_set_size,
                batch_size=batch_size,
            )
            # Get embeddings
            reference_embeddings, test_embeddings = get_embeddings(
                model=model,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                RESULT_PATH=RESULT_PATH,
                setting=setting,
                index_str=index_str,
            )

            print(
                "Loaded reference embeddings: ",
                reference_embeddings.shape,
                "test embeddings: ",
                test_embeddings.shape,
            )

            # Assert if embedding and label shape match
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

            # Memory cleanup
            torch.cuda.empty_cache()

            # Submit KNN jobs to process pool
            future = executor.submit(
                perform_knn,
                reference_embeddings=reference_embeddings,
                reference_labels=reference_labels,
                test_embeddings=test_embeddings,
                k=5,
                use_gpu=torch.cuda.is_available(),
            )

            future_to_model_dir[future] = model_dir

            # Memory cleanup
            del model, reference_embeddings, test_embeddings, future
            torch.cuda.empty_cache()
            gc.collect()

        for future in as_completed(future_to_model_dir):
            model_dir = future_to_model_dir[future]
            try:
                y_pred = future.result()
                update_pert_report(
                    y_pred_corr=y_pred,
                    y_true=test_labels,
                    model_dir=model_dir,
                    clf_report=clf_report,
                    RESULT_PATH=RESULT_PATH,
                    setting=setting,
                )
            except Exception as exc:
                print(f"{model_dir} generated an exception: {exc}")
            finally:
                print(f"Done evaluating {model_dir}")
                torch.cuda.empty_cache()
                gc.collect()
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

    print("Done.")

    # Evaluate random model
    if eval_random:
        # Check if Random has already been evaluated
        index_str = "Random"
        if index_str not in clf_report.index:
            print("Evaluating random model")
            evaluate_random_model(
                y_true=test_labels,
                gene_dim=gene_dim,
                train_set_size=train_set_size,
                val_set_size=val_set_size,
                batch_size=batch_size,
                clf_report=clf_report,
                RESULT_PATH=RESULT_PATH,
                reference_labels=reference_labels,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                setting=setting,
            )
