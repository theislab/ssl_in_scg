import torch
import anndata
import gc
import numpy as np
from self_supervision.estimator.cellnet import EstimatorAutoEncoder


def annotate_best_model_rec(
    adata: anndata.AnnData, estim: EstimatorAutoEncoder, ckpt_no_ssl: str, ckpt_ssl: str
):
    """
    Annotates the adata object with the information about the better model for each cell.

    Args:
        adata (anndata.AnnData): The AnnData object to be annotated.
        estim: The estimator object used for prediction.
        ckpt_no_ssl (str): The path to the checkpoint file of the model without self-supervised learning.
        ckpt_ssl (str): The path to the checkpoint file of the model with self-supervised learning.

    Returns:
        adata (anndata.AnnData): The annotated AnnData object.
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    estim_ssl = estim
    estim_no_ssl = estim

    # Load the models
    estim_no_ssl.model.load_state_dict(torch.load(ckpt_no_ssl)["state_dict"])
    estim_ssl.model.load_state_dict(torch.load(ckpt_ssl)["state_dict"])

    # Place models to device
    estim_no_ssl.model.to(device)
    estim_ssl.model.to(device)

    # Placeholder for storing the better model for each cell
    better_model_annotations = []

    # Iterate over the dataloader in batches
    for batch in estim.datamodule.test_dataloader():
        # Use batch_idx
        batch_idx = 0
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]
        # Send every tensor to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Compute reconstructed transcriptomes for both models
        x_reconst_no_ssl, _ = estim_no_ssl.model.predict_step(
            batch=batch, batch_idx=batch_idx
        )
        x_reconst_ssl, _ = estim_ssl.model.predict_step(
            batch=batch, batch_idx=batch_idx
        )

        # Iterate over each cell in the batch
        for reconst_no_ssl, reconst_ssl, true_x in enumerate(
            zip(x_reconst_no_ssl, x_reconst_ssl, batch["X"])
        ):
            # Compute the similarity between the reconstructed and actual transcriptomes
            similarity_no_ssl = compute_similarity(reconst_no_ssl, true_x)
            similarity_ssl = compute_similarity(reconst_ssl, true_x)

            # Determine which model is better
            better_model = "no_ssl" if similarity_no_ssl > similarity_ssl else "ssl"
            better_model_annotations.append(better_model)

        # Memory management
        del x_reconst_no_ssl, x_reconst_ssl
        torch.cuda.empty_cache()
        gc.collect()
        batch_idx += 1

    # Annotate the adata object with the information
    adata.obs["better_model"] = better_model_annotations

    return adata


def compute_similarity(reconstructed, actual):
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters:
    reconstructed (numpy.ndarray): The reconstructed array.
    actual (numpy.ndarray): The actual array.

    Returns:
    float: The Pearson correlation coefficient.
    """
    # Ensure the inputs are numpy arrays
    reconstructed = np.array(reconstructed.cpu().detach())
    actual = np.array(actual.cpu().detach())

    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(reconstructed, actual)

    # The correlation coefficient is the off-diagonal element in the 2x2 correlation matrix
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient
