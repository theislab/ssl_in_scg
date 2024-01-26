import numpy as np
import scanpy as sc
import anndata
import os
import shutil
import matplotlib.pyplot as plt


def load_data(embedding_path: str, labels: list):
    """
    Load data from the given embedding path and labels.

    Args:
        embedding_path (str): The path to the embedding file.
        labels (list): The list of labels.

    Returns:
        tuple: A tuple containing the embeddings and labels as numpy arrays.
    """
    embeddings = np.load(embedding_path)
    return embeddings, np.array(labels)


def create_subsampled_adata(embeddings, labels, num_samples=500000):
    """
    Create a subsampled AnnData object from the given embeddings and labels.

    Parameters:
    - embeddings (numpy.ndarray): The embeddings array of shape (n_samples, n_features).
    - labels (numpy.ndarray): The labels array of shape (n_samples,).
    - num_samples (int): The number of samples to subsample from the embeddings and labels. Default is 500000.

    Returns:
    - adata (anndata.AnnData): The subsampled AnnData object with the subsampled embeddings and labels.
    """
    total_samples = embeddings.shape[0]
    indices = np.random.choice(range(total_samples), num_samples, replace=False)
    subsampled_embeddings = embeddings[indices]
    subsampled_labels = labels[indices]

    adata = sc.AnnData(subsampled_embeddings)
    adata.obs["cell_type"] = subsampled_labels
    # Compute tSNE
    sc.tl.tsne(adata, use_rep="X")
    return adata


def plot_tsne(
    adata: anndata.AnnData,
    font_dict: dict,
    cell_freq_thres: int,
    save_dir: str,
    size: int = 5,
    title: str = "Zero-shot SSL",
):
    """
    Plot t-SNE visualization of cell types in the given AnnData object.
    """

    # Subset adata based on cell frequency, done in notebook
    # cell_freq = adata.obs["cell_type"].value_counts()
    # cells_to_keep = cell_freq[cell_freq > cell_freq_thres].index.tolist()

    colors = [
        '#d82626', '#d84126', '#d85b26', '#d87626', '#d89126', '#d8ac26', '#d8c626', 
        '#cfd826', '#b5d826', '#9ad826', '#7fd826', '#64d826', '#49d826', '#2fd826',
        '#26d838', '#26d852', '#26d86d', '#26d888', '#26d8a3', '#26d8bd', '#26d8d8', 
        '#26bdd8', '#26a3d8', '#2688d8', '#266dd8', '#2652d8', '#2638d8', '#2f26d8',
        '#4926d8', '#6426d8', '#7f26d8', '#9a26d8', '#b526d8', '#cf26d8', '#d826c6', 
        '#d826ac', '#d82691', '#d82676', '#d8265b', '#d82641'
    ]

    # Account for different capitalization in cell types, these are also the cell types to visualize
    # Index names
    """
    ['Neuron',
    'CD8-positive, Alpha-beta T Cell',
    'CD4-positive, Alpha-beta T Cell',
    'B Cell',
    'Classical Monocyte',
    'Natural Killer Cell',
    'T Cell',
    'Naive Thymus-derived CD4-positive, Alpha-beta T Cell',
    'Monocyte',
    'Oligodendrocyte',
    'Macrophage',
    'Central Memory CD4-positive, Alpha-beta T Cell',
    'Glutamatergic Neuron',
    'Endothelial Cell',
    'Fallopian Tube Secretory Epithelial Cell',
    'L2/3-6 Intratelencephalic Projecting Glutamatergic Cortical Neuron',
    'CD14-positive Monocyte',
    'Retinal Rod Cell',
    'Mature Alpha-beta T Cell',
    'Fibroblast Of Cardiac Tissue',
    'Cardiac Muscle Cell',
    'CD16-positive, CD56-dim Natural Killer Cell, Human',
    'Naive B Cell',
    'Luminal Epithelial Cell Of Mammary Gland',
    'Memory B Cell',
    'Alveolar Macrophage',
    'Plasma Cell',
    'CD14-positive, CD16-negative Classical Monocyte',
    'Basal Cell',
    'Naive Thymus-derived CD8-positive, Alpha-beta T Cell',
    'Epithelial Cell Of Proximal Tubule',
    'Enterocyte',
    'CD4-positive, Alpha-beta Memory T Cell',
    'Microglial Cell',
    'Type II Pneumocyte',
    'Non-classical Monocyte',
    'Regulatory T Cell',
    'Oligodendrocyte Precursor Cell',
    'Double-positive, Alpha-beta Thymocyte',
    'Effector Memory CD8-positive, Alpha-beta T Cell']
    """
    # Value names
    """
    ['neuron',
    'CD8-positive, alpha-beta T cell',
    'CD4-positive, alpha-beta T cell',
    'B cell',
    'classical monocyte',
    'natural killer cell',
    'T cell',
    'naive thymus-derived CD4-positive, alpha-beta T cell',
    'monocyte',
    'oligodendrocyte',
    'macrophage',
    'glutamatergic neuron',
    'central memory CD4-positive, alpha-beta T cell',
    'endothelial cell',
    'L2/3-6 intratelencephalic projecting glutamatergic cortical neuron',
    'fallopian tube secretory epithelial cell',
    'CD14-positive monocyte',
    'retinal rod cell',
    'cardiac muscle cell',
    'fibroblast of cardiac tissue',
    'mature alpha-beta T cell',
    'CD16-positive, CD56-dim natural killer cell, human',
    'luminal epithelial cell of mammary gland',
    'naive B cell',
    'memory B cell',
    'alveolar macrophage',
    'plasma cell',
    'CD14-positive, CD16-negative classical monocyte',
    'basal cell',
    'naive thymus-derived CD8-positive, alpha-beta T cell',
    'epithelial cell of proximal tubule',
    'enterocyte',
    'CD4-positive, alpha-beta memory T cell',
    'microglial cell',
    'type II pneumocyte',
    'non-classical monocyte',
    'double-positive, alpha-beta thymocyte',
    'regulatory T cell',
    'oligodendrocyte precursor cell',
    'CD8-positive, alpha-beta memory T cell']
    """
    cell_types_mapping = {
        'Neuron': 'neuron',
        'CD8-positive, Alpha-beta T Cell': 'CD8-positive, alpha-beta T cell',
        'CD4-positive, Alpha-beta T Cell': 'CD4-positive, alpha-beta T cell',
        'B Cell': 'B cell',
        'Classical Monocyte': 'classical monocyte',
        'Natural Killer Cell': 'natural killer cell',
        'T Cell': 'T cell',
        'Naive Thymus-derived CD4-positive, Alpha-beta T Cell': 'naive thymus-derived CD4-positive, alpha-beta T cell',
        'Monocyte': 'monocyte',
        'Oligodendrocyte': 'oligodendrocyte',
        'Macrophage': 'macrophage',
        'Central Memory CD4-positive, Alpha-beta T Cell': 'central memory CD4-positive, alpha-beta T cell',
        'Glutamatergic Neuron': 'glutamatergic neuron',
        'Endothelial Cell': 'endothelial cell',
        'Fallopian Tube Secretory Epithelial Cell': 'fallopian tube secretory epithelial cell',
        'L2/3-6 Intratelencephalic Projecting Glutamatergic Cortical Neuron': 'L2/3-6 intratelencephalic projecting glutamatergic cortical neuron',
        'CD14-positive Monocyte': 'CD14-positive monocyte',
        'Retinal Rod Cell': 'retinal rod cell',
        'Mature Alpha-beta T Cell': 'mature alpha-beta T cell',
        'Fibroblast Of Cardiac Tissue': 'fibroblast of cardiac tissue',
        'Cardiac Muscle Cell': 'cardiac muscle cell',
        'CD16-positive, CD56-dim Natural Killer Cell, Human': 'CD16-positive, CD56-dim natural killer cell, human',
        'Naive B Cell': 'naive B cell',
        'Luminal Epithelial Cell Of Mammary Gland': 'luminal epithelial cell of mammary gland',
        'Memory B Cell': 'memory B cell',
        'Alveolar Macrophage': 'alveolar macrophage',
        'Plasma Cell': 'plasma cell',
        'CD14-positive, CD16-negative Classical Monocyte': 'CD14-positive, CD16-negative classical monocyte',
        'Basal Cell': 'basal cell',
        'Naive Thymus-derived CD8-positive, Alpha-beta T Cell': 'naive thymus-derived CD8-positive, alpha-beta T cell',
        'Epithelial Cell Of Proximal Tubule': 'epithelial cell of proximal tubule',
        'Enterocyte': 'enterocyte',
        'CD4-positive, Alpha-beta Memory T Cell': 'CD4-positive, alpha-beta memory T cell',
        'Microglial Cell': 'microglial cell',
        'Type II Pneumocyte': 'type II pneumocyte',
        'Non-classical Monocyte': 'non-classical monocyte',
        'Regulatory T Cell': 'regulatory T cell',
        'Oligodendrocyte Precursor Cell': 'oligodendrocyte precursor cell',
        'Double-positive, Alpha-beta Thymocyte': 'double-positive, alpha-beta thymocyte',
        'Effector Memory CD8-positive, Alpha-beta T Cell': 'effector memory CD8-positive, alpha-beta T cell',
    }

    # Creating cells_to_keep and cells_to_keep_capitalized lists
    cells_to_keep = list(cell_types_mapping.values())
    cells_to_keep_capitalized = list(cell_types_mapping.keys())

    # Creating cell_type_to_color and cell_type_to_color_capitalized dictionaries
    cell_type_to_color = {ct: colors[i] for i, ct in enumerate(cells_to_keep)}
    cell_type_to_color_capitalized = {ct: colors[i] for i, ct in enumerate(cells_to_keep_capitalized)}

    # Hardcode
    if 'B Cell' in adata.obs["cell_type"].unique():
        cell_type_to_color = cell_type_to_color_capitalized
        cells_to_keep = cells_to_keep_capitalized
    elif 'B cell' in adata.obs["cell_type"].unique():
        cell_type_to_color = cell_type_to_color
        cells_to_keep = cells_to_keep
    else:
        raise ValueError(f"Cell types in adata.obs['cell_type'] are neither in cells_to_keep nor in cells_to_keep_capitalized. Cell types in adata.obs['cell_type']: {adata.obs['cell_type'].unique()}")

    adata_plot = adata[adata.obs["cell_type"].isin(cells_to_keep)].copy()

    # Calculate t-SNE if not already in adata
    if "X_tsne" not in adata_plot.obsm_keys():
        sc.tl.tsne(adata_plot, n_pcs=50, use_rep="X")

    # Set font parameters for the plot
    plt.rcParams["font.size"] = font_dict["fontsize"]
    plt.rcParams["font.family"] = font_dict["fontname"]

    # Plot t-SNE
    sc.pl.tsne(
        adata_plot,
        color="cell_type",
        title=title,
        show=True,
        size=size,
        legend_fontsize=font_dict["fontsize"],
        legend_fontoutline=1,
        save="dummy.png",
        palette=cell_type_to_color
    )

    # Move plot from current_dir + figures/tsnedummy.svg to save_dir
    file_path = os.path.join(save_dir, "tSNE_" + title + ".png")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_dir = os.getcwd()
    shutil.move(os.path.join(current_dir, "figures/tsnedummy.png"), file_path)
