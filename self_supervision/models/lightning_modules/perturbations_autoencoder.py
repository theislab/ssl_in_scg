# Autoencoder class as pretrained in self-supervision, but adapted to have the same functionality as scGEN for perturbation prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
import anndata
import lightning.pytorch as pl
from typing import Tuple, List, Callable, Literal
from torch.distributions import ContinuousBernoulli
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection
from torch.utils.data import DataLoader
import numpy as np
import gc
from self_supervision.models.base.base import MLP
from self_supervision.data.datamodules import AdataDataset


def extractor(
    data,
    cell_type,
    condition_key,
    cell_type_key,
    ctrl_key,
    stim_key,
):
    """
    Returns a list of `data` files while filtering for a specific `cell_type`.

    Parameters
    ----------
    data: `~anndata.AnnData`
        Annotated data matrix
    cell_type: basestring
        specific cell type to be extracted from `data`.
    condition_key: basestring
        key for `.obs` of `data` where conditions can be found.
    cell_type_key: basestring
        key for `.obs` of `data` where cell types can be found.
    ctrl_key: basestring
        key for `control` part of the `data` found in `condition_key`.
    stim_key: basestring
        key for `stimulated` part of the `data` found in `condition_key`.
    Returns
    ----------
    data_list: list
        list of `data` files while filtering for a specific `cell_type`.
    Example
    ----------
    ```python
    import scgen
    import anndata

    train_data = anndata.read("./data/train.h5ad")
    test_data = anndata.read("./data/test.h5ad")
    train_data_extracted_list = extractor(
        train_data, "CD4T", "conditions", "cell_type", "control", "stimulated"
    )
    ```
    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condition_1 = data[
        (data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == ctrl_key)
    ]
    condition_2 = data[
        (data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == stim_key)
    ]
    training = data[
        ~(
            (data.obs[cell_type_key] == cell_type)
            & (data.obs[condition_key] == stim_key)
        )
    ]
    return [training, condition_1, condition_2, cell_with_both_condition]


def balancer(
    adata,
    cell_type_key,
):
    """
    Makes cell type population equal.

    Parameters
    ----------
    adata: `~anndata.AnnData`
        Annotated data matrix.
    cell_type_key: basestring
        key for `.obs` of `data` where cell types can be found.
    Returns
    ----------
    balanced_data: `~anndata.AnnData`
        Equal cell type population Annotated data matrix.
    Example
    ----------
    ```python
    import scgen
    import anndata

    train_data = anndata.read("./train_kang.h5ad")
    train_ctrl = train_data[train_data.obs["condition"] == "control", :]
    train_ctrl = balancer(train_ctrl, "conditions", "cell_type")
    ```
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    index_all = []
    for cls in class_names:
        class_index = np.array(adata.obs[cell_type_key] == cls)
        index_cls = np.nonzero(class_index)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), max_number)]
        index_all.append(index_cls_r)

    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data


class SCGENAutoEncoder(pl.LightningModule):
    """
    This class implements the scGEN autoencoder.
    The scGEN autoencoder has the functionality of scGEN, but instead of relying on the VAE from scVI, it
    uses the same autoencoder structure as MLPAutoencoder in order to load the pretrained weights from SSL.
    """

    def __init__(
        self,
        # fixed params
        gene_dim: int,  # n_input in scGEN
        units_encoder: List[int],
        units_decoder: List[int],
        # scGEN params
        adata: anndata.AnnData,
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        # model specific params
        reconstruction_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        dropout: float = 0.1,
        activation: Callable[[], torch.nn.Module] = nn.SELU,
        output_activation: Callable[[], torch.nn.Module] = None,
    ):
        # check input
        assert 0.0 <= dropout <= 1.0
        assert reconstruction_loss in ["mse", "mae", "continuous_bernoulli", "bce"]

        self.reconst_loss = reconstruction_loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gc_freq = 5
        self.adata = adata

        super(SCGENAutoEncoder, self).__init__()

        self.encoder = MLP(
            in_channels=gene_dim,
            hidden_channels=units_encoder,
            activation_layer=activation,
            inplace=False,
            dropout=dropout,
        )
        # Define decoder network
        self.decoder = MLP(
            in_channels=units_encoder[-1],
            hidden_channels=units_decoder + [gene_dim],
            # norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
            activation_layer=activation,
            inplace=False,
            dropout=dropout,
        )

        metrics = MetricCollection(
            {
                "explained_var_weighted": ExplainedVariance(
                    multioutput="variance_weighted"
                ),
                "explained_var_uniform": ExplainedVariance(
                    multioutput="uniform_average"
                ),
                "mse": MeanSquaredError(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _calc_reconstruction_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ):
        """
        Calculate the reconstruction loss.

        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.
            reduction (str, optional): Reduction method. Defaults to 'mean'.

        Returns:
            torch.Tensor: Reconstruction loss.
        """
        if self.reconst_loss == "continuous_bernoulli":
            loss = -ContinuousBernoulli(probs=preds).log_prob(targets)
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
        elif self.reconst_loss == "bce":
            loss = F.binary_cross_entropy(preds, targets, reduction=reduction)
        elif self.reconst_loss == "mae":
            loss = F.l1_loss(preds, targets, reduction=reduction)
        else:
            loss = F.mse_loss(preds, targets, reduction=reduction)
        return loss

    def inference(self, x_in):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_latent = self.encoder(x_in)
        return x_latent

    def generative(self, x_latent):
        """
        High level generative method.
        Runs the generative (decoder) model.
        """
        x_reconst = self.decoder(x_latent)
        return x_reconst

    def forward(self, batch):
        """
        Forward pass of the autoencoder.

        Args:
            batch: Input batch.

        Returns:
            tuple: Tuple containing the latent representation and the reconstructed output.
        """
        x_in = batch["X"]
        x_latent = self.encoder(x_in)
        x_reconst = self.decoder(x_latent)
        # x_latent = self.inference(x_in)
        # x_reconst = self.generative(x_latent)
        return x_latent, x_reconst

    @torch.no_grad()
    def sample(
        self,
        batch,
        n_samples=1,
    ) -> np.ndarray:
        """
        Generate observation samples from the posterior predictive distribution.
        Parameters
        ----------
        batch
            Input batch.
        n_samples
            Number of required samples for each cell

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """

        inference_outputs, generative_outputs = self.forward(
            batch,
        )
        return generative_outputs

    def predict(
        self,
        ctrl_key=None,
        stim_key=None,
        adata_to_predict=None,
        celltype_to_predict=None,
        restrict_arithmetic_to="all",
        condition_key="condition",
        cell_type_key="celltype",
    ) -> anndata.AnnData:
        """
        Predicts the cell type provided by the user in stipulated condition.
        Analogue to scGEN.predict, but with autoencoder backbone.
        Parameters
        ----------
        ctrl_key: basestring
            key for `control` part of the `data` found in `condition_key`.
        stim_key: basestring
            key for `stimulated` part of the `data` found in `condition_key`.
        adata_to_predict: `~anndata.AnnData`
            Adata for unperturbed cells you want to be predicted.
        celltype_to_predict: basestring
            The cell type you want to be predicted.
        restrict_arithmetic_to: basestring or dict
            Dictionary of celltypes you want to be observed for prediction.
        Returns
        -------
        predicted_cells: np nd-array
            `np nd-array` of predicted cells in primary space.
        delta: float
            Difference between stimulated and control cells in latent space
        """
        if restrict_arithmetic_to == "all":
            ctrl_x = self.adata[self.adata.obs[condition_key] == ctrl_key, :]
            stim_x = self.adata[self.adata.obs[condition_key] == stim_key, :]
            ctrl_x = balancer(ctrl_x, cell_type_key)
            stim_x = balancer(stim_x, cell_type_key)
        else:
            key = list(restrict_arithmetic_to.keys())[0]
            values = restrict_arithmetic_to[key]
            subset = self.adata[self.adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[condition_key] == ctrl_key, :]
            stim_x = subset[subset.obs[condition_key] == stim_key, :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_type_key)
                stim_x = balancer(stim_x, cell_type_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception(
                "Please provide a cell type name or adata for your unperturbed cells"
            )
        if celltype_to_predict is not None:
            ctrl_pred = extractor(
                self.adata,
                celltype_to_predict,
                condition_key,
                cell_type_key,
                ctrl_key,
                stim_key,
            )[1]
        else:
            ctrl_pred = adata_to_predict

        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        ctrl_adata = ctrl_x[cd_ind, :]
        stim_adata = stim_x[stim_ind, :]

        latent_ctrl = self._avg_vector(ctrl_adata)
        latent_stim = self._avg_vector(stim_adata)

        delta = latent_stim - latent_ctrl  # np array

        latent_cd = self.get_latent_representation(ctrl_pred)

        stim_pred = delta + latent_cd
        predicted_cells = (
            self.generative(torch.Tensor(stim_pred)).cpu().detach().numpy()
        )

        predicted_adata = anndata.AnnData(
            X=predicted_cells,
            obs=ctrl_pred.obs.copy(),
            var=ctrl_pred.var.copy(),
            obsm=ctrl_pred.obsm.copy(),
        )
        return predicted_adata, delta

    def _avg_vector(self, adata):
        """
        Returns average latent vector for adata.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        Returns
            np.ndarray: Average latent vector for adata.
        ----------
        """
        return np.mean(self.get_latent_representation(adata), axis=0)

    def get_latent_representation(self, adata, batch_size=4096):
        """
        Returns latent representation of adata.
        """

        # Train and val dataset
        dataset = AdataDataset(
            genes=torch.tensor(adata.X.todense()),
            perturbations=adata.obs["perturbation"].values,
        )

        # Get dataloader
        num_cpus_available = 28  # os.cpu_count()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpus_available
        )

        # Get latent representation and concatenate it
        latent = []
        for batch in dataloader:
            # batch = auto_move_data(batch)
            latent.append(self.inference(batch["X"]))
        latent = torch.cat(latent, dim=0)
        return latent.cpu().detach().numpy()

    @abc.abstractmethod
    def _step(self, batch, training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method for performing a step in the training, validation, or testing process.

        Args:
            batch: Input batch data.
            training (bool, optional): Flag indicating whether the model is in training mode. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the predictions and the loss.
        """
        targets = batch["X"]
        # inputs = batch['X']

        # Forward pass
        latent, preds = self(batch)

        # Compute reconstruction loss
        reconst_loss = self._calc_reconstruction_loss(preds, targets)

        return preds, reconst_loss

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        x_reconst, loss = self._step(batch, training=True)
        self.log_dict(
            self.train_metrics(x_reconst, batch["X"]), on_epoch=True, on_step=True
        )
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        x_reconst, loss = self._step(batch, training=False)
        self.log_dict(self.val_metrics(x_reconst, batch["X"]))
        self.log("val_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        x_reconst, loss = self._step(batch, training=False)
        metrics = self.test_metrics(x_reconst, batch["X"])
        self.log_dict(metrics)
        self.log("test_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return metrics

    def configure_optimizers(self):
        """
        Configure optimizers.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def on_train_epoch_end(self) -> None:
        """
        Perform operations at the end of each training epoch.
        """
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        """
        Perform operations at the end of each validation epoch.
        """
        gc.collect()
