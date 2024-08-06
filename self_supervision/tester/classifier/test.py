import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import faiss
import dask.dataframe as dd
from self_supervision.tester.classifier.utils import (
    correct_labels,
    prepare_clf_report,
    get_index_str,
    update_clf_report,
    update_per_class_clf_report,
)
from self_supervision.tester.classifier.data_utils import (
    get_embeddings,
    load_and_process_labels,
    prepare_dataloader,
    prepare_data_path,
    initialize_subset,
    get_pca_embeddings,
    load_from_file,
)
from self_supervision.tester.classifier.model_utils import (
    load_and_wrap_model,
    prepare_estim,
)
from self_supervision.tester.classifier.evaluation_utils import (
    evaluate_knn_from_file,
    evaluate_random_model,
)
from self_supervision.tester.classifier.kNN import perform_knn
from self_supervision.paths import DATA_DIR, RESULTS_FOLDER
from typing import Optional, Tuple, Dict
import gc


def eval_emb_knn_per_class(
    estim,
    model_dirs: list[str],
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    test_dataloader: Optional[torch.utils.data.DataLoader] = None,
    setting: str = "CellNet",
) -> None:
    """
    Evaluate embeddings using k-nearest neighbors (k-NN) classification per class.

    Args:
        estim: The estimator object.
        model_dirs: A list of model directories.
        train_dataloader: Optional. The data loader for training data.
        val_dataloader: Optional. The data loader for validation data.
        test_dataloader: Optional. The data loader for test data.
        setting: Optional. The setting for evaluation. Must be one of ['CellNet', 'hlca', 'pbmc', 'tabula_sapiens', 'OOD_HiT', 'OOD_nn', 'OOD_Circ_Imm', 'OOD_Cort_Dev', 'OOD_Great_Apes'].

    Returns:
        None
    """

    # Assertions
    assert (
        setting
        in [
            "CellNet",
            "hlca",
            "pbmc",
            "tabula_sapiens",
            "OOD_HiT",
            "OOD_nn",
            "OOD_Circ_Imm",
            "OOD_Cort_Dev",
            "OOD_Great_Apes",
        ]
    ), f"Invalid setting. Must be one of CellNet, hlca, pbmc, tabula_sapiens, OOD_HiT, OOD_nn, OOD_Circ_Imm, OOD_Cort_Dev, OOD_Great_Apes. You provided: {setting}"

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(
        estim, train_dataloader, val_dataloader, test_dataloader
    )

    # Prepare data path:
    REFERENCE_PATH, DATA_PATH = prepare_data_path(DATA_DIR, setting)

    # Initialize subset if necessary
    supervised_subset = initialize_subset(setting)

    # Classification Report Path
    is_supervised = (
        True if "No_SSL" in model_dirs[0] else False
    )  # Artefact of handling lists
    if is_supervised:
        clf_report_path = os.path.join(
            RESULTS_FOLDER,
            "classification",
            "val_clf_per_class_report_"
            + str(supervised_subset)
            + "_supervised_knn.csv",
        )
    else:
        clf_report_path = os.path.join(
            RESULTS_FOLDER,
            "classification",
            "val_clf_per_class_report_" + str(supervised_subset) + "_ssl_knn.csv",
        )

    _, reference_labels, test_labels = load_and_process_labels(
        data_path=DATA_PATH,
        reference_path=REFERENCE_PATH,
        test_dataloader=test_dataloader,
        setting=setting,
    )

    # Evaluate models
    with ThreadPoolExecutor() as executor:
        future_to_model_dir = {}
        for model_dir in model_dirs:
            print(f"Evaluating {model_dir}")

            # Get index string for classification report
            index_str = get_index_str(model_dir)
            wrapped_model = load_and_wrap_model(
                model_dir=model_dir, estim=estim, supervised_subset=supervised_subset
            )
            estim = prepare_estim(estim, wrapped_model, batch_size=8192)

            # Get embeddingsf
            reference_embeddings, test_embeddings = get_embeddings(
                estim=estim,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                RESULT_PATH=RESULTS_FOLDER,
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
            del reference_embeddings, test_embeddings, future, wrapped_model
            torch.cuda.empty_cache()
            gc.collect()

        for future in as_completed(future_to_model_dir):
            model_dir = future_to_model_dir[future]
            try:
                y_pred = future.result()
                cell_type_hierarchy = np.load(
                    os.path.join(REFERENCE_PATH, "cell_type_hierarchy/child_matrix.npy")
                )
                y_pred_corr = correct_labels(test_labels, y_pred, cell_type_hierarchy)
                print(
                    "Evalute model ",
                    model_dir,
                    "generate is_supervised to ",
                    is_supervised,
                )
                # Update the classification report
                final_report = update_per_class_clf_report(
                    y_pred_corr=y_pred_corr,
                    y_true=test_labels,
                    model_dir=model_dir,
                    clf_report_path=clf_report_path,
                    supervised_subset=supervised_subset,
                    is_supervised=True if "No_SSL" in model_dir else False,
                )
            except Exception as exc:
                print(f"{model_dir} generated an exception: {exc}")
            finally:
                print(f"Done evaluating {model_dir}")
                torch.cuda.empty_cache()
                gc.collect()
            # Memory cleanup
            del y_pred, y_pred_corr, cell_type_hierarchy
            torch.cuda.empty_cache()
            gc.collect()

    # Load and filter test labels
    test_labels_reference = dd.read_parquet(
        os.path.join(DATA_DIR, "test"), columns=["dataset_id"]
    )
    test_labels = dd.read_parquet(
        os.path.join(DATA_DIR, "test"), columns=["cell_type"]
    )
    test_labels_filtered = (
        test_labels[test_labels_reference["dataset_id"] == supervised_subset]
        .compute()
        .to_numpy()
    )

    # Calculate cell counts and map them to cell types
    cell_types, cell_counts = np.unique(test_labels_filtered, return_counts=True)

    # Convert cell counts to integer
    cell_counts = cell_counts.astype(int)

    # Create a dictionary mapping cell types to cell counts
    cell_count_dict = dict(zip(cell_types, cell_counts))

    # Ensure that the indices of final_report are integers (if they are not already)
    final_report.index = final_report.index.astype(int)

    # Map cell counts to the report
    final_report["Cell Count"] = final_report.index.map(cell_count_dict)

    # Remove rows with NaN in 'Cell Count'
    final_report = final_report.dropna(subset=["Cell Count"])

    # Check for any NaNs in 'Cell Count' and handle if necessary
    nan_count = final_report["Cell Count"].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in 'Cell Count'.")

    # Add 'Cell Type' column based on the index
    cell_type_mapping = pd.read_parquet(
        os.path.join(DATA_DIR, "categorical_lookup/cell_type.parquet")
    )
    mapped_labels = cell_type_mapping.iloc[
        final_report.index.astype(int)
    ].values.flatten()
    final_report["Cell Type"] = mapped_labels

    # Add F1 differences
    # final_report['F1 Difference'] = final_report['Self-Supervised F1'] - final_report['Supervised F1']

    # Save the final report
    final_report.to_csv(clf_report_path)

    print("Done.")

    return final_report


def get_total_correct(
    estim,
    model_dirs: list[str],
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    test_dataloader: Optional[torch.utils.data.DataLoader] = None,
    setting: str = "CellNet",
    batch_size: int = 8192,
) -> Dict[str, int]:
    """
    Evaluate the total number of correct predictions per cell type for each model in model_dirs.

    Parameters:
    - estim: The estimator object used for prediction.
    - model_dirs: A list of directories containing the models to be evaluated.
    - train_dataloader: The dataloader for the training data.
    - val_dataloader: The dataloader for the validation data.
    - test_dataloader: The dataloader for the test data.
    - setting: The setting for evaluation. Must be one of ['CellNet', 'hlca', 'pbmc', 'tabula_sapiens', 'OOD_HiT', 'OOD_nn', 'OOD_Circ_Imm', 'OOD_Cort_Dev', 'OOD_Great_Apes'].
    - batch_size: The batch size for evaluation.

    Returns:
    - A dictionary containing the total number of correct predictions per cell type for each model in model_dirs.
    """
    # Assertions
    assert (
        setting
        in [
            "CellNet",
            "hlca",
            "pbmc",
            "tabula_sapiens",
            "OOD_HiT",
            "OOD_nn",
            "OOD_Circ_Imm",
            "OOD_Cort_Dev",
            "OOD_Great_Apes",
        ]
    ), f"Invalid setting. Must be one of CellNet, hlca, pbmc, tabula_sapiens, OOD_HiT, OOD_nn, OOD_Circ_Imm, OOD_Cort_Dev, OOD_Great_Apes. You provided: {setting}"

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(
        estim, train_dataloader, val_dataloader, test_dataloader
    )

    # Prepare data path:
    REFERENCE_PATH, DATA_PATH = prepare_data_path(DATA_DIR, setting)

    # Initialize subset if necessary
    supervised_subset = initialize_subset(setting)

    # Evaluate models
    with ThreadPoolExecutor() as executor:
        future_to_model_dir = {}
        for model_dir in model_dirs:
            print(f"Evaluating {model_dir}")

            # Get index string for classification report
            index_str = get_index_str(model_dir)

            _, reference_labels, test_labels = load_and_process_labels(
                data_path=DATA_PATH,
                reference_path=REFERENCE_PATH,
                test_dataloader=test_dataloader,
                setting=setting,
            )

            wrapped_model = load_and_wrap_model(
                model_dir=model_dir, estim=estim, supervised_subset=supervised_subset
            )
            estim = prepare_estim(estim, wrapped_model, batch_size=8192)

            # Get embeddingsf
            reference_embeddings, test_embeddings = get_embeddings(
                estim=estim,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                RESULT_PATH=RESULTS_FOLDER,
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
            del reference_embeddings, test_embeddings, future, wrapped_model
            torch.cuda.empty_cache()
            gc.collect()

        for future in as_completed(future_to_model_dir):
            model_dir = future_to_model_dir[future]
            index_str = get_index_str(model_dir)
            try:
                # _, reference_labels, test_labels = load_and_process_labels(
                #     data_path=DATA_PATH,
                #     reference_path=REFERENCE_PATH,
                #     test_dataloader=test_dataloader,
                #     setting=setting,
                # )

                y_pred = future.result()
                print('y_pred', y_pred.shape)
                print('test_labels', test_labels.shape)
                cell_type_hierarchy = np.load(
                    os.path.join(REFERENCE_PATH, "cell_type_hierarchy/child_matrix.npy")
                )
                y_pred_corr = correct_labels(test_labels, y_pred, cell_type_hierarchy)

                # Save the predicted labels to disk
                np.save(
                    os.path.join(
                        RESULTS_FOLDER,
                        "classification",
                        "new_predicted_labels_" + index_str + ".npy",
                    ),
                    y_pred_corr,
                )
                print(
                    "Saved predicted labels to disk at ",
                    os.path.join(
                        RESULTS_FOLDER,
                        "classification",
                        "new_predicted_labels_" + index_str + ".npy",
                    ),
                )

                # Save true labels to disk
                np.save(
                    os.path.join(
                        RESULTS_FOLDER,
                        "classification",
                        "new_true_labels_" + index_str + ".npy",
                    ),
                    test_labels,
                )
                print(
                    "Saved true labels to disk at ",
                    os.path.join(
                        RESULTS_FOLDER,
                        "classification",
                        "new_true_labels_" + index_str + ".npy",
                    ),
                )

                # Without the mapping
                correct_counts = pd.DataFrame(
                    index=np.unique(test_labels),
                    columns=["Correct Count"],
                    dtype=int,
                )

                # Count the number of correct predictions (from y_pred_corr vs. mapped_labels)
                for label in np.unique(test_labels):
                    correct_counts.loc[label, "Correct Count"] = int(
                        np.sum(y_pred_corr[test_labels == label])
                    )

                # Save the counts to disk
                correct_counts.to_csv(
                    os.path.join(
                        RESULTS_FOLDER,
                        "classification",
                        "new_correct_counts_" + index_str + ".csv",
                    )
                )
                print(
                    "Saved correct counts to disk at ",
                    os.path.join(
                        RESULTS_FOLDER,
                        "classification",
                        "new_correct_counts_" + index_str + ".csv",
                    ),
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

    # Save total number of true counts per cell type to disk
    test_labels_reference = dd.read_parquet(
        os.path.join(DATA_DIR, "test"), columns=["dataset_id"]
    )
    test_labels = dd.read_parquet(
        os.path.join(DATA_DIR, "test"), columns=["cell_type"]
    )
    cell_type_mapping = pd.read_parquet(
        os.path.join(DATA_DIR, "categorical_lookup/cell_type.parquet")
    )

    test_labels_filtered = (
        test_labels[test_labels_reference["dataset_id"] == supervised_subset]
        .compute()
        .to_numpy()
        .flatten()
    )

    # Map labels
    mapped_labels = cell_type_mapping.loc[test_labels_filtered, "label"].values

    # Unique labels for indexing the DataFrame
    unique_labels = np.unique(mapped_labels)

    # Create DataFrame for storing the counts of correct predictions per cell type
    true_counts = pd.DataFrame(index=unique_labels, columns=["True Count"], dtype=int)

    # Count the number of correct predictions (from y_pred_corr vs. mapped_labels)
    for label in unique_labels:
        true_counts.loc[label, "True Count"] = np.sum(mapped_labels == label)

    # Save the counts to disk
    true_counts.to_csv(os.path.join(RESULTS_FOLDER, "classification", "true_counts.csv"))
    print(
        "Saved true counts to disk at ",
        os.path.join(RESULTS_FOLDER, "classification", setting + "_true_counts.csv"),
    )


def eval_emb_knn(
    estim,
    model_dirs: list[str],
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    test_dataloader: Optional[torch.utils.data.DataLoader] = None,
    setting: str = "CellNet",
    eval_random: bool = True,
    eval_pca: bool = True,
    eval_file: Optional[Tuple[str, str, str, str]] = None,
    batch_size: int = 8192,
) -> None:
    # Assertions
    assert (
        setting
        in [
            "CellNet",
            "hlca",
            "pbmc",
            "tabula_sapiens",
            "OOD_HiT",
            "OOD_nn",
            "OOD_Circ_Imm",
            "OOD_Cort_Dev",
            "OOD_Great_Apes",
        ]
    ), f"Invalid setting. Must be one of CellNet, hlca, pbmc, tabula_sapiens, OOD_HiT, OOD_nn. You provided: {setting}"
    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(
        estim, train_dataloader, val_dataloader, test_dataloader
    )

    # Prepare data path:
    REFERENCE_PATH, DATA_PATH = prepare_data_path(DATA_DIR, setting)

    # Initialize subset if necessary
    supervised_subset = initialize_subset(setting)

    # Load classification report
    clf_report = prepare_clf_report(RESULT_PATH=RESULTS_FOLDER, setting=setting)

    _, reference_labels, test_labels = load_and_process_labels(
        data_path=DATA_PATH,
        reference_path=REFERENCE_PATH,
        test_dataloader=test_dataloader,
        setting=setting,
    )
    # Initialize FAISS GPU resources
    res = None
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()  # Initialize GPU resources

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
            wrapped_model = load_and_wrap_model(
                model_dir=model_dir, estim=estim, supervised_subset=supervised_subset
            )
            estim = prepare_estim(estim, wrapped_model, batch_size=8192)

            # Get embeddings
            reference_embeddings, test_embeddings = get_embeddings(
                estim=estim,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                RESULT_PATH=RESULTS_FOLDER,
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
            del wrapped_model
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

            del estim.model

            # Memory cleanup
            del reference_embeddings, test_embeddings, future
            torch.cuda.empty_cache()
            gc.collect()

        for future in as_completed(future_to_model_dir):
            model_dir = future_to_model_dir[future]
            try:
                y_pred = future.result()
                cell_type_hierarchy = np.load(
                    os.path.join(REFERENCE_PATH, "cell_type_hierarchy/child_matrix.npy")
                )
                y_pred_corr = correct_labels(test_labels, y_pred, cell_type_hierarchy)
                update_clf_report(
                    y_pred_corr=y_pred_corr,
                    y_true=test_labels,
                    model_dir=model_dir,
                    clf_report=clf_report,
                    RESULT_PATH=RESULTS_FOLDER,
                    setting=setting,
                )
            except Exception as exc:
                print(f"{model_dir} generated an exception: {exc}")
            finally:
                print(f"Done evaluating {model_dir}")
                torch.cuda.empty_cache()
                gc.collect()
            # Memory cleanup
            del y_pred, y_pred_corr, cell_type_hierarchy
            torch.cuda.empty_cache()
            gc.collect()

    print("Done.")

    # Evaluate random model
    if eval_random:
        # Check if Random has already been evaluated
        index_str = "Random"
        if index_str not in clf_report.index:
            cell_type_hierarchy = np.load(
                os.path.join(REFERENCE_PATH, "cell_type_hierarchy/child_matrix.npy")
            )
            print("Evaluating random model")
            evaluate_random_model(
                estim=estim,
                y_true=test_labels,
                clf_report=clf_report,
                RESULT_PATH=RESULTS_FOLDER,
                cell_type_hierarchy=cell_type_hierarchy,
                reference_labels=reference_labels,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                setting=setting,
                batch_size=batch_size,
            )

    # Evaluate PCA
    if eval_pca:
        print("Evaluating PCA")
        # Check if PCA has already been evaluated
        index_str = "PCA"
        if index_str not in clf_report.index:
            cell_type_hierarchy = np.load(
                os.path.join(REFERENCE_PATH, "cell_type_hierarchy/child_matrix.npy")
            )

            # Load PCA embeddings
            reference_embeddings, test_embeddings = get_pca_embeddings(
                DATA_PATH=DATA_PATH, setting=setting
            )

            evaluate_knn_from_file(
                reference_embeddings=reference_embeddings,
                test_embeddings=test_embeddings,
                reference_labels=reference_labels,
                test_labels=test_labels,
                cell_type_hierarchy=cell_type_hierarchy,
                clf_report=clf_report,
                RESULT_PATH=RESULTS_FOLDER,
                res=res,
                setting=setting,
                k=5,
                index_str=index_str,
            )

    # Evaluate from file
    if eval_file is not None:
        print("Evaluating from file")
        reference_embeddings, reference_labels, test_embeddings, test_labels = load_from_file(
            eval_file=eval_file
        )

        index_str = "PCA_Test"  # Hardcode for now

        if index_str not in clf_report.index:
            cell_type_hierarchy = np.load(
                os.path.join(REFERENCE_PATH, "cell_type_hierarchy/child_matrix.npy")
            )

            evaluate_knn_from_file(
                reference_embeddings=reference_embeddings,
                test_embeddings=test_embeddings,
                reference_labels=reference_labels,
                test_labels=test_labels,
                cell_type_hierarchy=cell_type_hierarchy,
                clf_report=clf_report,
                RESULT_PATH=RESULTS_FOLDER,
                res=res,
                setting=setting,
                k=5,
                index_str=index_str,
            )
