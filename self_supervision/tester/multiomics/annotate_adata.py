import torch
import anndata
import gc
import numpy as np
import os
import scanpy as sc
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from self_supervision.models.lightning_modules.multiomics_autoencoder import (
    MultiomicsAutoencoder,
)
from self_supervision.data.datamodules import MultiomicsDataloader
from self_supervision.trainer.multiomics.multiomics_utils import one_hot_encode


def annotate_multiomics_adata(
    adata: anndata.AnnData,
    ckpt_no_ssl: str,
    ckpt_ssl: str,
    batch_size: int = 1028,
    setting: str = "id_test",
):
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if setting == "id_test":
        test_ds = MultiomicsDataloader(
            np.log1p(adata[adata.obs["split"] == "test"].obsm["protein_counts"]),
            adata[adata.obs["split"] == "test"].X.todense(),
            one_hot_encode(adata[adata.obs["split"] == "test"].obs["batch"]),
        )

    elif setting == "ood_test":
        test_ds = MultiomicsDataloader(
            np.log1p(adata[adata.obs["split"] == "ood_test"].obsm["protein_counts"]),
            adata[adata.obs["split"] == "ood_test"].X.todense(),
            one_hot_encode(adata[adata.obs["split"] == "ood_test"].obs["batch"]),
        )
    else:
        raise ValueError("Invalid setting. Must be either id_test or ood_test.")

    # Create DataLoaders
    cpus = os.cpu_count()
    num_workers = max(1, cpus // 2)  # Ensure at least one worker is used

    test_dl = DataLoader(
        test_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )  # Updated batch_size from 1 to args.batch_size

    # Load the models
    supervised_model = MultiomicsAutoencoder(
        mode="fine_tuning",
        model="MAE",
        dropout=0.11642113240634665,
        learning_rate=0.00011197711341004587,
        weight_decay=0.0010851761758488817,
        batch_size=batch_size,
        model_type="autoencoder",  # Standard is 'autoencoder'
    )
    ssl_model = MultiomicsAutoencoder(
        mode="fine_tuning",
        model="MAE",
        dropout=0.11642113240634665,
        learning_rate=0.00011197711341004587,
        weight_decay=0.0010851761758488817,
        batch_size=batch_size,
        model_type="autoencoder",  # Standard is 'autoencoder'
    )

    checkpoint_supervised = torch.load(ckpt_no_ssl)
    checkpoint_ssl = torch.load(ckpt_ssl)
    # print('checkpoint: ', checkpoint)
    # If checkpoint is a normal dict and state_dict is a key, then load the state_dict
    if (
        isinstance(checkpoint_supervised, dict)
        and "state_dict" in checkpoint_supervised
    ):
        supervised_dict = {
            k: v
            for k, v in checkpoint_supervised["state_dict"].items()
            if ("decoder" not in k) and ("projector" not in k)
        }
    # If checkpoint is a OrderedDict then the keys are in OrderedDict([('Key1', tensor([...]), ('Key2', tensor([...]))])
    elif isinstance(checkpoint_supervised, OrderedDict):
        supervised_dict = {
            k: v
            for k, v in checkpoint_supervised.items()
            if ("decoder" not in k) and ("projector" not in k)
        }
    if isinstance(checkpoint_ssl, dict) and "state_dict" in checkpoint_ssl:
        ssl_dict = {
            k: v
            for k, v in checkpoint_ssl["state_dict"].items()
            if ("decoder" not in k) and ("projector" not in k)
        }
    elif isinstance(checkpoint_ssl, OrderedDict):
        ssl_dict = {
            k: v
            for k, v in checkpoint_ssl.items()
            if ("decoder" not in k) and ("projector" not in k)
        }

    supervised_model.load_state_dict(supervised_dict, strict=False)
    ssl_model.load_state_dict(ssl_dict, strict=False)

    # Place models to device
    supervised_model.to(device)
    ssl_model.to(device)
    batch_idx = 0
    # Iterate over test dataloader
    for batch in test_dl:
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]
        # Send every tensor to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Compute reconstructed transcriptomes for both models
        x_latent_no_ssl, x_reconst_no_ssl = supervised_model.predict_step(
            batch=batch, batch_idx=batch_idx
        )
        x_latent_ssl, x_reconst_ssl = ssl_model.predict_step(
            batch=batch, batch_idx=batch_idx
        )

        # Concatenate the latent space and reconstructed transcriptomes
        if batch_idx == 0:
            x_latent_no_ssl_all = x_latent_no_ssl
            x_reconst_no_ssl_all = x_reconst_no_ssl
            x_latent_ssl_all = x_latent_ssl
            x_reconst_ssl_all = x_reconst_ssl
        else:
            x_latent_no_ssl_all = torch.cat(
                (x_latent_no_ssl_all, x_latent_no_ssl), dim=0
            )
            x_reconst_no_ssl_all = torch.cat(
                (x_reconst_no_ssl_all, x_reconst_no_ssl), dim=0
            )
            x_latent_ssl_all = torch.cat((x_latent_ssl_all, x_latent_ssl), dim=0)
            x_reconst_ssl_all = torch.cat((x_reconst_ssl_all, x_reconst_ssl), dim=0)

        batch_idx += 1

    # Add the latent space and reconstructed transcriptomes to the adata object
    adata.obsm["X_latent_supervised"] = x_latent_no_ssl_all.cpu().detach().numpy()
    adata.obsm["Proteins_supervised"] = x_reconst_no_ssl_all.cpu().detach().numpy()
    adata.obsm["X_latent_self_supervised"] = x_latent_ssl_all.cpu().detach().numpy()
    adata.obsm["Proteins_self_supervised"] = x_reconst_ssl_all.cpu().detach().numpy()

    # Add UMAPs
    if "X_umap" not in adata.obsm.keys():
        sc.tl.umap(adata)

    return adata


def plot_protein_predictions(adata, font, num_proteins: int = 10):
    plt.rc("font", **font)

    protein_names = list(adata.obsm["protein_counts"].columns)

    print("Protein names:")

    # Sort proteins by correlation
    correlations = []
    for protein in protein_names:
        true_counts = adata.obsm["protein_counts"][protein].values
        supervised_counts = np.expm1(
            adata.obsm["Proteins_supervised"][:, protein_names.index(protein)]
        )
        self_supervised_counts = np.expm1(
            adata.obsm["Proteins_self_supervised"][:, protein_names.index(protein)]
        )

        corr_supervised = pearsonr(true_counts, supervised_counts)[0]
        corr_self_supervised = pearsonr(true_counts, self_supervised_counts)[0]

        correlations.append((protein, corr_supervised, corr_self_supervised))

    correlations = sorted(correlations, key=lambda x: x[2], reverse=True)

    fig, axes = plt.subplots(num_proteins, 3, figsize=(10, 10))

    print("Correlations: ", correlations[:num_proteins])

    for i, (protein, corr_supervised, corr_self_supervised) in enumerate(
        correlations[:num_proteins]
    ):
        true_counts = adata.obsm["protein_counts"][protein].values
        supervised_counts = np.expm1(
            adata.obsm["Proteins_supervised"][:, protein_names.index(protein)]
        )
        self_supervised_counts = np.expm1(
            adata.obsm["Proteins_self_supervised"][:, protein_names.index(protein)]
        )

        # Add to adata_test.obs
        adata.obs[f"{protein}_supervised"] = supervised_counts
        adata.obs[f"{protein}_self_supervised"] = self_supervised_counts

        corr_supervised = pearsonr(true_counts, supervised_counts)[0]
        corr_self_supervised = pearsonr(true_counts, self_supervised_counts)[0]

        sc.pl.umap(
            adata, color=protein, ax=axes[i, 0], show=False, title=f"{protein} True"
        )
        sc.pl.umap(
            adata,
            color=f"{protein}_supervised",
            ax=axes[i, 1],
            show=False,
            title=f"Supervised (r={corr_supervised:.2f})",
        )
        sc.pl.umap(
            adata,
            color=f"{protein}_self_supervised",
            ax=axes[i, 2],
            show=False,
            title=f"Self-Supervised (r={corr_self_supervised:.2f})",
        )

    plt.tight_layout()
    for ax in axes.flat:
        ax.title.set_fontsize(5)
        ax.xaxis.label.set_fontsize(5)
        ax.yaxis.label.set_fontsize(5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(5)

    plt.show()


def old_annotate_multiomics_adata(
    adata: anndata.AnnData,
    estim: MultiomicsAutoencoder,
    ckpt_no_ssl: str,
    ckpt_ssl: str,
):
    """
    Annotates the adata object with latent space and reconstructed protein transcriptomes
    using a multiomics autoencoder model.

    Args:
        adata (anndata.AnnData): The AnnData object to be annotated.
        estim (MultiomicsAutoencoder): The multiomics autoencoder model.
        ckpt_no_ssl (str): Path to the checkpoint file for the model without self-supervised learning.
        ckpt_ssl (str): Path to the checkpoint file for the model with self-supervised learning.

    Returns:
        adata (anndata.AnnData): The annotated AnnData object.
    """

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Place models to device
    estim.model.to(device)
    estim.model.mode = "fine_tuning"

    # Placeholder for storing the better model for each cell
    estim_ssl = estim
    estim_no_ssl = estim

    # Load the models
    estim_no_ssl.model.load_state_dict(torch.load(ckpt_no_ssl)["state_dict"])
    estim_ssl.model.load_state_dict(torch.load(ckpt_ssl)["state_dict"])

    # Iterate over the dataloader in batches, first for supervised
    for batch in estim.datamodule.test_dataloader():
        # Use batch_idx
        batch_idx = 0
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]
        # Send every tensor to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Compute reconstructed transcriptomes for both models
        x_latent_no_ssl, x_reconst_no_ssl = estim_no_ssl.model.predict_step(
            batch=batch, batch_idx=batch_idx
        )
        # Concatenate the latent space and reconstructed transcriptomes
        if batch_idx == 0:
            x_latent_no_ssl_all = x_latent_no_ssl
            x_reconst_no_ssl_all = x_reconst_no_ssl
        else:
            x_latent_no_ssl_all = torch.cat(
                (x_latent_no_ssl_all, x_latent_no_ssl), dim=0
            )
            x_reconst_no_ssl_all = torch.cat(
                (x_reconst_no_ssl_all, x_reconst_no_ssl), dim=0
            )

    # Add the latent space and reconstructed transcriptomes to the adata object
    adata.obsm["X_latent_supervised"] = x_latent_no_ssl_all.cpu().detach().numpy()
    adata.obsm["Proteins_supervised"] = x_reconst_no_ssl_all.cpu().detach().numpy()

    # Memory management
    del x_latent_no_ssl_all, x_reconst_no_ssl_all
    torch.cuda.empty_cache()
    gc.collect()

    # Iterate over the dataloader in batches, now for unsupervised
    for batch in estim.datamodule.test_dataloader():
        # Use batch_idx
        batch_idx = 0
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]
        # Send every tensor to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Compute reconstructed transcriptomes for both models
        x_latent_ssl, x_reconst_ssl = estim_ssl.model.predict_step(
            batch=batch, batch_idx=batch_idx
        )
        # Concatenate the latent space and reconstructed transcriptomes
        if batch_idx == 0:
            x_latent_ssl_all = x_latent_ssl
            x_reconst_ssl_all = x_reconst_ssl
        else:
            x_latent_ssl_all = torch.cat((x_latent_ssl_all, x_latent_ssl), dim=0)
            x_reconst_ssl_all = torch.cat((x_reconst_ssl_all, x_reconst_ssl), dim=0)

    # Add the latent space and reconstructed transcriptomes to the adata object
    adata.obsm["X_latent_self_supervised"] = x_latent_ssl_all.cpu().detach().numpy()
    adata.obsm["Proteins_self_supervised"] = x_reconst_ssl_all.cpu().detach().numpy()

    # Memory management
    del x_latent_ssl_all, x_reconst_ssl_all
    torch.cuda.empty_cache()
    gc.collect()

    return adata
