import torch
import pandas as pd
import anndata
import os
import pickle
from typing import Tuple, List, Dict
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from self_supervision.models.lightning_modules.multiomics_autoencoder import (
    MultiomicsCiteAutoencoder,
    MultiomicsMultiAutoencoder,
    FG_BG_MultiomicsAutoencoder,
)
from self_supervision.paths import RESULTS_FOLDER


def spearman_corr(x, y):
    return spearmanr(x, y).correlation


def evaluate_all(
        observed: pd.DataFrame,
        predictions: Dict[str, pd.DataFrame],
        metrics=["pearson", "spearman", "mse"],
        gene_dim=2000,
        n_proteins=134,

) -> pd.DataFrame:
    """
    Evaluate the performance of predictions.
    Create a dataframe with "Model", "Pearson Correlation", "Spearman Correlation", "MSE", "Model Type" columns.
    For each model (key in predictions), calculate the Pearson correlation, Spearman correlation, and MSE between the observed and predicted proteins (entire matrix, flatten)
    Append the results to the dataframe.
    Args:
        observed (pd.DataFrame): Ground truth data.
        predictions (Dict[str, pd.DataFrame]): Model predictions.
        metrics (list): List of metrics to compute ("pearson", "spearman", "mse").
        gene_dim (int): Number of gene dimensions, default is 2000.
        n_proteins (int): Number of proteins, default is 134.

    Returns:
        pd.DataFrame: Evaluation results for each cell and overall mean.
    """
    if predictions is None:
        raise ValueError("No predictions provided.")
    results = []

    for model_name, prediction in predictions.items():
        prediction_df = pd.DataFrame(prediction, index=observed.index)
        if prediction_df.shape[1] > n_proteins:
            predicted_values = prediction_df.iloc[:, gene_dim : gene_dim + n_proteins]
        else:
            predicted_values = prediction_df

        # Flatten the arrays for correlation calculations
        observed_values = observed.values.flatten()
        predicted_values = predicted_values.values.flatten()
        row_result = {
            "Model": model_name
        }
        if "pearson" in metrics:
            row_result["Pearson Correlation"] = pearsonr(observed_values, predicted_values)[0]
        if "spearman" in metrics:
            row_result["Spearman Correlation"] = spearman_corr(observed_values, predicted_values)
        if "mse" in metrics:
            row_result["MSE"] = mean_squared_error(observed_values, predicted_values)
        results.append(row_result)

        
    results_df = pd.DataFrame(results)
    return results_df  

def evaluate_per_cell(
    observed: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    metrics=["pearson", "spearman", "mse"],
    difficult=False,
    gene_dim=2000,
    n_proteins=134,
    threshold=0.9
) -> pd.DataFrame:
    """
    Evaluate the performance of predictions per cell.

    Args:
        observed (pd.DataFrame): Ground truth data.
        predictions (Dict[str, pd.DataFrame]): Model predictions.
        metrics (list): List of metrics to compute ("pearson", "spearman", "mse").
        difficult (bool): Whether to focus on difficult cells.
        threshold (float): Threshold for defining easy cells based on Pearson correlation.

    Returns:
        pd.DataFrame: Evaluation results for each cell and overall mean.
    """
    if predictions is None:
        raise ValueError("No predictions provided.")
    results = []
    all_cells_difficult = pd.DataFrame(index=observed.index)

    for model_name, prediction in predictions.items():
        prediction_df = pd.DataFrame(prediction, index=observed.index)
        for i in range(len(prediction_df)):
            observed_values = observed.iloc[i, :]
            if prediction_df.shape[1] > n_proteins:
                predicted_values = prediction_df.iloc[i, gene_dim : gene_dim + n_proteins]
            else:
                predicted_values = prediction_df.iloc[i, :]
            row_result = {
                "Model": model_name,
                "Cell": observed.index[i]
            }
            if "pearson" in metrics:
                row_result["Pearson Correlation"] = pearsonr(observed_values, predicted_values)[0]
            if "spearman" in metrics:
                row_result["Spearman Correlation"] = spearman_corr(observed_values, predicted_values)
            if "mse" in metrics:
                row_result["MSE"] = mean_squared_error(observed_values, predicted_values)
            results.append(row_result)
    
    results_df = pd.DataFrame(results)
    
    # Identify difficult cells
    if difficult:
        for cell in observed.index:
            correlations = results_df[results_df["Cell"] == cell]["Pearson Correlation"]
            all_cells_difficult.loc[cell, "is_difficult"] = (correlations < threshold).all()
        difficult_cells = all_cells_difficult[all_cells_difficult["is_difficult"]].index
        results_df = results_df[results_df["Cell"].isin(difficult_cells)]

        print("Kept ", len(results_df), "/", len(predictions) * len(observed.index), " cells") 
    
    mean_results_df = results_df.groupby("Model").mean().reset_index()
    return results_df, mean_results_df

def evaluate_per_protein(
    observed: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    gene_dim: int = 2000,
    n_proteins: int = 134,
    metrics=["pearson", "spearman", "mse"],
    difficult=False,
    threshold=0.9
) -> pd.DataFrame:
    """
    Evaluate the performance of predictions per protein.

    Args:
        observed (pd.DataFrame): Ground truth data.
        predictions (Dict[str, pd.DataFrame]): Model predictions.
        gene_dim (int): Number of gene dimensions, default is 2000.
        n_proteins (int): Number of proteins, default is 134.
        metrics (list): List of metrics to compute ("pearson", "spearman", "mse").
        difficult (bool): Whether to focus on difficult proteins.
        threshold (float): Threshold for defining easy proteins based on Pearson correlation.

    Returns:
        pd.DataFrame: Evaluation results for each protein and overall mean.
    """
    if predictions is None:
        raise ValueError("No predictions provided.")
    results = []
    all_proteins_difficult = pd.DataFrame(index=observed.columns)

    for model_name, prediction in predictions.items():
        prediction_df = pd.DataFrame(prediction[:, gene_dim : gene_dim + n_proteins], columns=observed.columns)
        for protein in observed.columns:
            observed_values = observed[protein]
            predicted_values = prediction_df[protein]
            row_result = {
                "Model": model_name,
                "Protein": protein
            }
            if "pearson" in metrics:
                row_result["Pearson Correlation"] = pearsonr(observed_values, predicted_values)[0]
            if "spearman" in metrics:
                row_result["Spearman Correlation"] = spearman_corr(observed_values, predicted_values)
            if "mse" in metrics:
                row_result["MSE"] = mean_squared_error(observed_values, predicted_values)
            results.append(row_result)
    
    results_df = pd.DataFrame(results)
    
    # Identify difficult proteins
    if difficult:
        for protein in observed.columns:
            correlations = results_df[results_df["Protein"] == protein]["Pearson Correlation"]
            all_proteins_difficult.loc[protein, "is_difficult"] = (correlations < threshold).all()
        difficult_proteins = all_proteins_difficult[all_proteins_difficult["is_difficult"]].index
        results_df = results_df[results_df["Protein"].isin(difficult_proteins)]

        print("Kept ", len(results_df), "/", len(predictions) * len(observed.columns), " proteins")
    
    mean_results_df = results_df.groupby("Model").mean().reset_index()
    return results_df, mean_results_df


def rename_model(model_name):
    if "No_SSL" in model_name:
        return "Supervised"
    elif "linear" in model_name:
        return "Linear"
    elif "SSL_Random_Mask" in model_name and "20M" in model_name:
        return "Self-Supervised:\nRandom Mask\nscTab Dataset"
    elif "SSL_Random_Mask" in model_name and "NeurIPS" in model_name:
        return "Self-Supervised\nRandom Mask\nNeurIPS Dataset"
    elif "totalVI" in model_name:
        return "totalVI"
    
    else:
        return model_name
    

def get_pred(
    model_path: str,
    adata: anndata.AnnData,
    test_dl: torch.utils.data.DataLoader,
    model: str = "MAE",
    dropout: float = 0.11642113240634665,
    learning_rate: float = 0.00011197711341004587,
    weight_decay: float = 0.0010851761758488817,
    batch_size: int = 256,
    omics: str = "cite",  # "multi"
    gene_dim: int = 2000,
    n_proteins: int = 134,
    n_batches=12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns protein predictions and embeddings for a given model and test data.

    Args:
    - model_path (str): path to the saved model
    - adata (anndata.AnnData): AnnData object containing the test data
    - test_dl (torch.utils.data.DataLoader): DataLoader object containing the test data
    - model (str): type of model to use, default is 'MAE'
    - dropout (float): dropout rate, default is 0.11642113240634665
    - learning_rate (float): learning rate, default is 0.00011197711341004587
    - weight_decay (float): weight decay, default is 0.0010851761758488817
    - batch_size (int): batch size, default is 256
    - model_type (str): type of model, default is 'autoencoder'
    - gene_dim (int): number of gene dimensions, default is 2000
    - n_proteins (int): number of proteins, default is 134

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: protein predictions and embeddings
    """
    if model == "MAE":
        # Get model
        model = torch.load(model_path)

        # Get model type
        if omics == "cite":
            ae_model = MultiomicsCiteAutoencoder(
                mode="fine_tuning",
                model=model,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
            )

        # Load model
        ae_model.load_state_dict(model["state_dict"], strict=False)
        ae_model.eval()

        # Get predictions
        all_predictions = None
        all_embs = None
        with torch.no_grad():
            for batch in test_dl:
                gene = batch["X"]
                protein = batch["protein"]
                covariate = batch["batch"]
                gene = gene.squeeze(1)
                mask_all_protein = torch.zeros_like(protein)
                inputs = torch.cat((gene, mask_all_protein, covariate), dim=1)
                embs, predictions = ae_model(inputs)
                all_predictions = (
                    predictions
                    if all_predictions is None
                    else torch.cat((all_predictions, predictions), dim=0)
                )
                all_embs = (
                    embs if all_embs is None else torch.cat((all_embs, embs), dim=0)
                )

    elif model == "NegBin":
        # Get model
        model = torch.load(model_path)

        # Get model type
        negbin_model = FG_BG_MultiomicsAutoencoder(
            batch_size=batch_size,
        )

        # Load model
        negbin_model.load_state_dict(model["state_dict"], strict=False)
        negbin_model.eval()

        # Get predictions
        all_predictions = None
        all_embs = None
        with torch.no_grad():
            for batch in test_dl:
                # Extract Gene, Protein, and Covariate data, Zero out Protein and Covariate data
                gene = batch["X"]
                protein = (
                    batch["protein"]
                    if "protein" in batch
                    else torch.zeros(gene.shape[0], n_proteins).to(gene.device)
                )
                in_proteins = torch.zeros(gene.shape[0], n_proteins).to(
                    gene.device
                )  # Model input proteins are zeroed
                covariate = (
                    batch["batch"]
                    if "batch" in batch
                    else torch.zeros(gene.shape[0], n_batches).to(gene.device)
                )
                in_covariates = torch.zeros(gene.shape[0], n_batches).to(
                    gene.device
                )  # Model input covariates are zeroed

                # Ensure gene is 2D
                if gene.dim() == 3:
                    gene = gene.squeeze(1)
                if gene.shape[-1] == gene_dim:
                    pass  # correct gene shape
                elif gene.shape[-1] > gene_dim:
                    # Load gene indices if necessary
                    root = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                    )
                    if omics == "multi":
                        multiomics_indices = pickle.load(
                            open(
                                root + "/self_supervision/data/multiomics_multi_indices.pickle",
                                "rb",
                            )
                        )
                    else:
                        multiomics_indices = pickle.load(
                            open(
                                root + "/self_supervision/data/multiomics_cite_indices.pickle",
                                "rb",
                            )
                        )
                    # Select only the multiomics indices
                    gene = gene[:, multiomics_indices]
                else:
                    raise ValueError(f"Unsupported gene shape: {gene.shape}")

                inputs = torch.cat((gene, in_proteins, in_covariates), dim=1)
                (
                    x_latent,
                    fg_mu,
                    fg_theta,
                    bg_mu,
                    bg_theta,
                    protein_preds,
                ) = negbin_model(inputs)
                all_predictions = (
                    protein_preds
                    if all_predictions is None
                    else torch.cat((all_predictions, protein_preds), dim=0)
                )
                all_embs = (
                    x_latent
                    if all_embs is None
                    else torch.cat((all_embs, x_latent), dim=0)
                )

    return all_predictions, all_embs


def get_pred_dir(
    model_dirs: List[str],
    adata: anndata.AnnData,
    test_dl: torch.utils.data.DataLoader,
    dropout: float = 0.11642113240634665,
    learning_rate: float = 0.00011197711341004587,
    weight_decay: float = 0.0010851761758488817,
    batch_size: int = 256,
    model_type: str = "autoencoder",
    gene_dim: int = 2000,
    n_proteins: int = 134,
    omics: str = "cite",
) -> Dict[str, pd.DataFrame]:
    """
    Wraps get_pred() for multiple models and returns a dictionary containing the protein predictions and embeddings for each model.

    Args:
    model_dirs (List[str]): A list of model paths.

    Returns:
    dict: A dictionary containing the protein predictions and embeddings for each model.
    """

    all_protein_predictions = {}
    all_embs = {}
    for model_dir in model_dirs:
        model = "NegBin" if "NegBin" in model_dir else "MAE"
        model_name = model_dir.split("/")[-5]
        print(f"Getting predictions for {model_name}...")
        protein_predictions, embs = get_pred(
            model_path=model_dir,
            adata=adata,
            test_dl=test_dl,
            model=model,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            omics=omics,
            gene_dim=gene_dim,
            n_proteins=n_proteins,
        )
        all_protein_predictions[model_name] = protein_predictions
        all_embs[model_name] = embs
    return all_protein_predictions, all_embs


def plot_results(mean_results_df, color_supervised, color_ssl, color_baseline, save_flag=False, include_linear=True, save_path="", setting=""):
    # Rename models
    mean_results_df['Model Type'] = mean_results_df['Model'].apply(rename_model)

    # Define the order of models
    if include_linear:
        model_order = ["Linear", "totalVI", "Supervised", "Self-Supervised\nRandom Mask\nNeurIPS Dataset", "Self-Supervised:\nRandom Mask\nscTab Dataset"]
    else:
        model_order = ["totalVI", "Supervised", "Self-Supervised\nRandom Mask\nNeurIPS Dataset", "Self-Supervised:\nRandom Mask\nscTab Dataset"]

    # Define color mapping for each model type
    model_colors = {
        "Supervised": color_supervised,
        "Self-Supervised:\nRandom Mask\nscTab Dataset": color_ssl,
        "Self-Supervised\nRandom Mask\nNeurIPS Dataset": color_ssl,
        "totalVI": color_baseline,
        "Linear": color_baseline,
    }

    # Define font properties
    font = {'family': 'sans-serif', 'size': 5}  # This will be for titles and labels

    # Box plot for Pearson Correlation
    plt.figure(figsize=(4, 2.5))
    ax1 = sns.boxplot(x='Model Type', y='Pearson Correlation', data=mean_results_df, linewidth=0.5, order=model_order, palette=model_colors)
    ax1.set_xlabel('Model Type', fontdict=font)
    ax1.set_ylabel('Pearson Correlation', fontdict=font)
    ax1.set_title('Cross-Modality Prediction (Pearson)', fontdict=font)

    # Set font for all tick labels to match the fontdict
    tick_font = {'fontsize': 5, 'fontname': 'sans-serif'}
    ax1.set_xticklabels(ax1.get_xticklabels(), **tick_font)
    ax1.set_yticklabels(ax1.get_yticklabels(), **tick_font)
    if save_flag:
        # Save as SVG with setting
        plt.savefig(f"{save_path}/cite_pearson_corr_{setting}.svg", bbox_inches='tight')
    plt.show()

    # Box plot for Spearman Correlation
    plt.figure(figsize=(4, 2.5))
    ax2 = sns.boxplot(x='Model Type', y='Spearman Correlation', data=mean_results_df, linewidth=0.5, order=model_order, palette=model_colors)
    ax2.set_xlabel('Model Type', fontdict=font)
    ax2.set_ylabel('Spearman Correlation', fontdict=font)
    ax2.set_title('Cross-Modality Prediction (Spearman)', fontdict=font)

    # Set font for all tick labels to match the fontdict
    ax2.set_xticklabels(ax2.get_xticklabels(), **tick_font)
    ax2.set_yticklabels(ax2.get_yticklabels(), **tick_font)
    if save_flag:
        plt.savefig(f"{save_path}/cite_spearman_corr_{setting}.svg", bbox_inches='tight')
    plt.show()

    # Box plot for MSE
    plt.figure(figsize=(4, 2.5))
    ax3 = sns.boxplot(x='Model Type', y='MSE', data=mean_results_df, linewidth=0.5, order=model_order, palette=model_colors)
    ax3.set_xlabel('Model Type', fontdict=font)
    ax3.set_ylabel('MSE', fontdict=font)
    ax3.set_title('Cross-Modality Prediction (MSE)', fontdict=font)

    # Set font for all tick labels to match the fontdict
    ax3.set_xticklabels(ax3.get_xticklabels(), **tick_font)
    ax3.set_yticklabels(ax3.get_yticklabels(), **tick_font)
    if save_flag:
        plt.savefig(f"{save_path}/cite_mse_{setting}.svg", bbox_inches='tight')
    plt.show()


def plot_protein_umaps(
    adata_test: anndata.AnnData,
    all_embs: Dict[str, torch.Tensor],
    all_protein_predictions: Dict[str, torch.Tensor],
    selected_proteins: List[str],
    num_genes: int = 2000,
    num_proteins: int = 134,
    figsize: Tuple[int, int] = (15, 10),
    fontdict: Dict[str, str] = {"fontsize": 14, "family": "sans-serif"},
):
    """
    Plots UMAPs for a list of proteins, showing true counts, supervised, and self-supervised predictions.

    Parameters:
    adata_test (AnnData): The AnnData object containing the test dataset.
    all_embs (dict): Dictionary of embeddings for supervised and self-supervised models.
    all_protein_predictions (dict): Dictionary of protein predictions from different models.
    selected_proteins (list): List of proteins to plot.
    num_genes (int): Number of genes.
    num_proteins (int): Number of proteins.
    figsize (tuple): Size of the figure.
    fontdict (dict): Font properties for plot titles and labels.
    """

    # Prepare AnnData objects
    supervised_embs = anndata.AnnData(
        X=all_embs["New_No_SSL_run0"].numpy(), obs=adata_test.obs
    )
    self_supervised_embs = anndata.AnnData(
        X=all_embs["SSL_Random_Mask_20Mrun8"].numpy(), obs=adata_test.obs
    )

    # Compute UMAPs
    sc.pp.neighbors(supervised_embs, use_rep="X")
    sc.pp.neighbors(self_supervised_embs, use_rep="X")
    sc.tl.umap(supervised_embs)
    sc.tl.umap(self_supervised_embs)

    # Determine the number of rows for subplots
    n_rows = len(selected_proteins)

    # Create a figure with subplots
    fig, axs = plt.subplots(n_rows, 3, figsize=figsize)

    for i, selected_protein in enumerate(selected_proteins):
        # True protein counts
        adata_test.obs[selected_protein] = np.log1p(
            adata_test.obsm["protein_counts"][selected_protein].values
        )
        adata_test.obsm["X_umap"] = adata_test.obsm["GEX_X_umap"]

        sc.pl.umap(
            adata_test,
            color=selected_protein,
            ax=axs[i, 0],
            show=False,
            title=f"True Counts - {selected_protein}",
        )

        # Supervised predictions
        protein_idx = adata_test.obsm["protein_counts"].columns.get_loc(
            selected_protein
        )
        protein_preds = all_protein_predictions["New_No_SSL_run0"][
            :, num_genes : num_genes + num_proteins
        ][:, protein_idx]
        supervised_embs.obs[selected_protein] = protein_preds

        pearson_corr_supervised = pearsonr(
            adata_test.obs[selected_protein], protein_preds
        )[0]
        sc.pl.umap(
            supervised_embs,
            color=selected_protein,
            ax=axs[i, 1],
            show=False,
            title=f"Supervised - {selected_protein} (Corr: {pearson_corr_supervised:.2f})",
        )

        # Self-supervised predictions
        protein_preds = all_protein_predictions["SSL_Random_Mask_20Mrun8"][
            :, num_genes : num_genes + num_proteins
        ][:, protein_idx]
        self_supervised_embs.obs[selected_protein] = protein_preds

        pearson_corr_self_supervised = pearsonr(
            adata_test.obs[selected_protein], protein_preds
        )[0]
        sc.pl.umap(
            self_supervised_embs,
            color=selected_protein,
            ax=axs[i, 2],
            show=False,
            title=f"Self-Supervised - {selected_protein} (Corr: {pearson_corr_self_supervised:.2f})",
        )

    for ax in axs.flat:
        # Adjust font of plot title and labels
        ax.title.set_fontsize(fontdict["fontsize"])
        ax.xaxis.label.set_fontsize(fontdict["fontsize"])
        ax.yaxis.label.set_fontsize(fontdict["fontsize"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fontdict["fontsize"])
            label.set_family(fontdict["family"])

        # Adjust spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Get the colorbar from the Heatmap
        cbar = ax.collections[0].colorbar

        # Ensure the colorbar ticks are visible
        cbar.ax.tick_params(labelsize=fontdict["fontsize"])

        # Change the colorbar axis font size and thickness
        cbar.ax.yaxis.label.set_fontsize(fontdict["fontsize"])
        cbar.ax.yaxis.label.set_fontweight("bold")
        cbar.outline.set_linewidth(0.5)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(
        RESULTS_FOLDER + "/multiomics/UMAPs.svg", bbox_inches="tight"
    )  # Save as SVG
    plt.show()


def compute_correlations_and_prepare_data(
    adata, supervised_key, self_supervised_key, protein_count_key, protein_names
):
    results_list = []

    # Loop over each protein to calculate the correlation
    for protein in protein_names:
        observed = adata.obsm[protein_count_key][protein].values
        supervised = adata.obsm[supervised_key][
            :, adata.obsm[protein_count_key].columns.get_loc(protein)
        ]
        self_supervised = adata.obsm[self_supervised_key][
            :, adata.obsm[protein_count_key].columns.get_loc(protein)
        ]

        # Filter out non-positive values for log transformation
        mask = (observed > 0) & (supervised > 0) & (self_supervised > 0)
        observed_log = np.log1p(observed[mask])
        supervised_log = np.log1p(supervised[mask])
        self_supervised_log = np.log1p(self_supervised[mask])

        # Calculate correlations, handle cases where arrays might be empty due to filtering
        corr_supervised = (
            pearsonr(observed_log, supervised_log)[0]
            if observed_log.size > 0
            else np.nan
        )
        corr_self_supervised = (
            pearsonr(observed_log, self_supervised_log)[0]
            if observed_log.size > 0
            else np.nan
        )

        # Store the results
        results_list.extend([(protein, "Observed", val) for val in observed_log])
        results_list.extend(
            [(protein, "Supervised", val, corr_supervised) for val in supervised_log]
        )
        results_list.extend(
            [
                (protein, "Self-Supervised", val, corr_self_supervised)
                for val in self_supervised_log
            ]
        )

    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(
        results_list, columns=["Protein", "Type", "Log Counts", "Correlation"]
    )
    return results_df


def plot_protein_correlations(results_df):
    # Set the aesthetic parameters of the plots
    sns.set_theme(context="notebook", style="whitegrid")

    # Convert the results DataFrame into long-form
    long_df = results_df.pivot(
        index="Protein", columns="Type", values="Log Counts"
    ).reset_index()
    long_df.columns.name = None  # Remove the index name

    # Merge the correlations into the long-form DataFrame
    long_df = long_df.merge(
        results_df.drop_duplicates(subset=["Protein", "Type"])[
            ["Protein", "Type", "Correlation"]
        ],
        on=["Protein", "Type"],
        how="left",
    )

    # Create the scatter plots
    g = sns.FacetGrid(long_df, col="Protein", col_wrap=5, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="Observed", y="Supervised", label="Supervised")
    g.map_dataframe(
        sns.scatterplot, x="Observed", y="Self-Supervised", label="Self-Supervised"
    )

    # Adjust the figure size
    g.fig.set_size_inches(15, 10)
    plt.subplots_adjust(top=0.9)

    # Set the title
    g.fig.suptitle("Imputed vs Denoised Correlations", fontsize=16)

    # Remove the axes titles for a cleaner look
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
    g.set_axis_labels("", "")

    # Add legend
    plt.legend()

    plt.show()


def plot_heatmap(protein_predictions: pd.DataFrame, observed: pd.DataFrame):
    """
    Plots a heatmap of protein predictions and observed values.

    Args:
    protein_predictions (pd.DataFrame): DataFrame containing protein predictions.
    observed (pd.DataFrame): DataFrame containing observed values.

    Returns:
    None
    """
    df = protein_predictions
    df2 = pd.DataFrame(observed)
    df.columns = df2.columns
    vmin = min(df.values.min(), df2.values.min())
    vmax = max(df.values.max(), df2.values.max())

    fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[1, 1, 0.1]))

    sns.heatmap(df, annot=False, cbar=False, ax=axs[0], vmax=vmax, vmin=vmin)
    sns.heatmap(df2, annot=False, yticklabels=False, cbar=False, ax=axs[1])

    fig.colorbar(axs[0].collections[0], cax=axs[2])
    plt.show()



def calc_metrics(predicted_proteins_expression: torch.Tensor, normalized_proteins: np.ndarray):
    """
    Calculate the metrics (MSE, Pearson Corr, Spearman Corr) for the predicted protein expression and the normalized protein expression.
    1) Average over all cells
    2) 90% Threshold over all cells
    3) Average over all proteins
    4) 90% Threshold over all proteins
    
    Input (both normalized)
    - predicted_proteins_expression: Predicted protein expression from the model, shape (n_cells, n_proteins)
    - normalized_proteins: Normalized protein expression, shape (n_cells, n_proteins)
    """

    results = {}

    # Calculate metrics for all cells
    pearson_cells = []
    spearman_cells = []
    mse_cells = []

    for i in range(normalized_proteins.shape[0]):
        observed_values = normalized_proteins[i, :]
        predicted_values = predicted_proteins_expression[i, :]
        pearson_corr = pearsonr(observed_values, predicted_values)[0]
        spearman_corr = spearmanr(observed_values, predicted_values)[0]
        mse = mean_squared_error(observed_values, predicted_values)
        pearson_cells.append(pearson_corr)
        spearman_cells.append(spearman_corr)
        mse_cells.append(mse)

    results["All_Cells"] = {
        "Pearson Correlation": np.mean(pearson_cells),
        "Spearman Correlation": np.mean(spearman_cells),
        "MSE": np.mean(mse_cells)
    }

    # Calculate 90% threshold metrics for cells
    threshold = 0.9
    difficult_cells_mask = np.array(pearson_cells) < threshold
    results["Difficult_Cells"] = {
        "Pearson Correlation": np.mean(np.array(pearson_cells)[difficult_cells_mask]),
        "Spearman Correlation": np.mean(np.array(spearman_cells)[difficult_cells_mask]),
        "MSE": np.mean(np.array(mse_cells)[difficult_cells_mask])
    }

    # Calculate metrics for all proteins
    pearson_proteins = []
    spearman_proteins = []
    mse_proteins = []

    for i in range(normalized_proteins.shape[1]):
        observed_values = normalized_proteins[:, i]
        predicted_values = predicted_proteins_expression[:, i]
        pearson_corr = pearsonr(observed_values, predicted_values)[0]
        spearman_corr = spearmanr(observed_values, predicted_values)[0]
        mse = mean_squared_error(observed_values, predicted_values)
        pearson_proteins.append(pearson_corr)
        spearman_proteins.append(spearman_corr)
        mse_proteins.append(mse)

    results["All_Proteins"] = {
        "Pearson Correlation": np.mean(pearson_proteins),
        "Spearman Correlation": np.mean(spearman_proteins),
        "MSE": np.mean(mse_proteins)
    }

    # Calculate 90% threshold metrics for proteins
    difficult_proteins_mask = np.array(pearson_proteins) < threshold
    results["Difficult_Proteins"] = {
        "Pearson Correlation": np.mean(np.array(pearson_proteins)[difficult_proteins_mask]),
        "Spearman Correlation": np.mean(np.array(spearman_proteins)[difficult_proteins_mask]),
        "MSE": np.mean(np.array(mse_proteins)[difficult_proteins_mask])
    }

    return results