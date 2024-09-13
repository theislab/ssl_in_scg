import abc
import gc
from typing import Callable, Dict, List, Optional, Tuple
import os
import numpy as np
import pickle
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, ContinuousBernoulli
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection
from torchmetrics.classification import MulticlassF1Score
from self_supervision.models.base.base import MLP
from self_supervision.models.contrastive.byol import BYOL
from self_supervision.models.contrastive.bt2 import (
    BarlowTwins,
    Transform,
    LARS,
    adjust_learning_rate,
)


def _mask_gene_programs_numpy(
    inputs: torch.Tensor, encoded_gene_program: np.ndarray, masking_rate: float
):
    """
    Randomly choose masking_rate percent of the gene programs (rows) to mask.
    Mask all the genes in the inputs tensor that are 1s in the chosen row of encoded_gene_program.
    return a binary mask of the tensor and a fraction of how many genes are set to 0.
    :param inputs: input data
    :param encoded_gene_program: encoded gene programs
    :param masking_rate: rate of gene programs to mask
    :return: mask, frac: mask is a binary mask of the tensor and frac is a fraction of how many genes are set to 0.
    """
    # Randomly choose masking_rate percent of the gene programs to mask
    num_gene_programs = encoded_gene_program.shape[0]
    num_gene_programs_to_mask = int(num_gene_programs * masking_rate)
    gene_programs_to_mask = np.random.permutation(num_gene_programs)[
        :num_gene_programs_to_mask
    ]
    # Mask the input tensor and cast it to numpy on cpu
    mask = np.ones_like(inputs.cpu().numpy())
    for ix in gene_programs_to_mask:
        mask[:, encoded_gene_program[ix, :] == 1] = 0
    return mask, 1 - mask.sum() / mask.size


def _only_activate_gene_program_numpy(
    inputs: torch.Tensor,
    # write that encoded_gene_program is a dict of a tuple of torch.Tensor
    # encoded_gene_program: Dict[Tuple[torch.Tensor, torch.Tensor]],
    encoded_gene_program: Dict,
):
    """
    For each row of the inputs tensor randomly choose one transcription factor (key of encoded_gene_program)
    The first element of encoded_gene_program[key] is a vector (torch.Tensor) of indices of genes in the gene program.
    The input mask should correspond to that mask, i.e., the mask is 1 where the genes are in the gene program.
    The second element of encoded_gene_program[key] is a vector (torch.Tensor) of index of the corresponding gene program.
    The output mask should be 1 that vector is 1.
    :param inputs: input data
    :param encoded_gene_program: encoded gene programs
    :return: (input_mask, output_mask): input_mask is a binary mask of the tensor and output_mask is a binary mask of the tensor.
    """
    # Randomly choose one transcription factor (key of encoded_gene_program) per row of the inputs tensor
    num_gene_programs = len(encoded_gene_program)
    gene_programs_to_activate = np.random.permutation(num_gene_programs)[
        : inputs.shape[0]
    ]
    # Mask the input tensor and cast it to numpy on cpu
    input_mask = np.zeros_like(inputs.cpu().numpy())
    output_mask = np.zeros_like(inputs.cpu().numpy())
    for ix, key in enumerate(gene_programs_to_activate):
        input_mask[ix, encoded_gene_program[key][0]] = 1
        output_mask[ix, encoded_gene_program[key][1]] = 1
    return input_mask, output_mask


def _mask_single_gene_programs(
    inputs: torch.Tensor,
    encoded_gene_program: np.ndarray,
):
    """
    Randomly choose 1 gene program, mask everything apart from that gene program both in the input and output.
    """
    # Randomly choose 1 gene program
    num_gene_programs = encoded_gene_program.shape[0]
    gene_program_to_mask = np.random.permutation(num_gene_programs)[:1]
    # Mask the input tensor and cast it to numpy on cpu
    input_mask = np.zeros_like(inputs.cpu().numpy())
    output_mask = np.zeros_like(inputs.cpu().numpy())
    input_mask[:, encoded_gene_program[gene_program_to_mask, :] == 1] = 1
    output_mask[:, encoded_gene_program[gene_program_to_mask, :] == 1] = 1
    return input_mask, output_mask


class BaseAutoEncoder(pl.LightningModule, abc.ABC):
    autoencoder: nn.Module  # autoencoder mapping von gene_dim to gene_dim

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        # params from datamodule
        batch_size: int,
        # model specific params
        reconst_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        gc_frequency: int = 5,
        automatic_optimization: bool = True,
    ):
        super(BaseAutoEncoder, self).__init__()

        self.automatic_optimization = automatic_optimization

        self.gene_dim = gene_dim
        self.batch_size = batch_size
        self.gc_freq = gc_frequency

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.reconst_loss = reconst_loss
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

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

    @abc.abstractmethod
    def _step(self, batch):
        """Calculate predictions (int64 tensor) and loss"""
        pass

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     batch = batch[0]
    #     return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict):  # Case for MultiomicsDataloader
            return batch
        else:
            return batch[0]

    def forward(self, batch):
        x_in = batch["X"]  # zero center data
        # do not use covariates
        x_latent = self.encoder(x_in)
        x_reconst = self.decoder(x_latent)
        return x_latent, x_reconst

    def on_train_epoch_end(self) -> None:
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        gc.collect()

    def configure_optimizers(self):
        optimizer_config = {
            "optimizer": self.optim(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        }
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = (
                {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            )
            interval = lr_scheduler_kwargs.pop("interval", "epoch")
            monitor = lr_scheduler_kwargs.pop("monitor", "val_loss_epoch")
            frequency = lr_scheduler_kwargs.pop("frequency", 1)
            scheduler = self.lr_scheduler(
                optimizer_config["optimizer"], **lr_scheduler_kwargs
            )
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": monitor,
                "frequency": frequency,
            }

        return optimizer_config


class BaseClassifier(pl.LightningModule, abc.ABC):
    classifier: Callable  # classifier mapping von gene_dim to type_dim - outputs logits

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        class_weights: np.ndarray,
        child_matrix: np.ndarray,
        # params from datamodule
        batch_size: int,
        # model specific params
        supervised_subset: Optional[int] = None,
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        gc_frequency: int = 1,
    ):
        super(BaseClassifier, self).__init__()

        self.gene_dim = gene_dim
        self.type_dim = type_dim
        self.batch_size = batch_size
        self.gc_freq = gc_frequency
        self.supervised_subset = supervised_subset

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        metrics = MetricCollection(
            {
                "f1_micro": MulticlassF1Score(num_classes=type_dim, average="micro"),
                "f1_macro": MulticlassF1Score(num_classes=type_dim, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.register_buffer("class_weights", torch.tensor(class_weights.astype("f4")))
        self.register_buffer("child_lookup", torch.tensor(child_matrix.astype("i8")))


    @abc.abstractmethod
    def _step(self, batch, training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def hierarchy_correct(self, preds, targets) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_is_child_node_or_node = (
            torch.sum(
                self.child_lookup[targets, :] * F.one_hot(preds, self.type_dim), dim=1
            )
            > 0
        )

        return (
            torch.where(pred_is_child_node_or_node, targets, preds),  # corrected preds
            torch.where(
                pred_is_child_node_or_node, preds, targets
            ),  # corrected targets
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        with torch.no_grad():
            batch = batch[0]
            batch["cell_type"] = torch.squeeze(batch["cell_type"])

        return batch

    def forward(self, x: torch.Tensor):
        if self.multiomics_indices:
            # If x is a tuple, which can happen in some testing due to lightning, then we need to extract the tensor
            if isinstance(x, tuple):
                x = x[0]["X"]
            x = x[:, self.multiomics_indices]
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}
        preds, loss = self._step(batch, training=True)
        self.log("train_loss", loss)
        f1_macro = self.train_metrics["f1_macro"](preds, batch["cell_type"])
        f1_micro = self.train_metrics["f1_micro"](preds, batch["cell_type"])
        self.log("train_f1_macro_step", f1_macro)
        self.log("train_f1_micro_step", f1_micro)

        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}
        preds, loss = self._step(batch, training=False)
        self.log("val_loss", loss)
        self.val_metrics["f1_macro"].update(preds, batch["cell_type"])
        self.val_metrics["f1_micro"].update(preds, batch["cell_type"])
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}
        preds, loss = self._step(batch, training=False)
        self.log("test_loss", loss)
        self.test_metrics["f1_macro"].update(preds, batch["cell_type"])
        self.test_metrics["f1_micro"].update(preds, batch["cell_type"])
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def on_train_epoch_end(self) -> None:
        self.log("train_f1_macro_epoch", self.train_metrics["f1_macro"].compute())
        self.train_metrics["f1_macro"].reset()
        self.log("train_f1_micro_epoch", self.train_metrics["f1_micro"].compute())
        self.train_metrics["f1_micro"].reset()
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        f1_macro = self.val_metrics["f1_macro"].compute()
        self.log("val_f1_macro", f1_macro)
        self.log("hp_metric", f1_macro)
        self.val_metrics["f1_macro"].reset()
        self.log("val_f1_micro", self.val_metrics["f1_micro"].compute())
        self.val_metrics["f1_micro"].reset()
        gc.collect()

    def on_test_epoch_end(self) -> None:
        self.log("test_f1_macro", self.test_metrics["f1_macro"].compute())
        self.test_metrics["f1_macro"].reset()
        self.log("test_f1_micro", self.test_metrics["f1_micro"].compute())
        self.test_metrics["f1_micro"].reset()
        gc.collect()

    def configure_optimizers(self):
        optimizer_config = {
            "optimizer": self.optim(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        }
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = (
                {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            )
            interval = lr_scheduler_kwargs.pop("interval", "epoch")
            monitor = lr_scheduler_kwargs.pop("monitor", "val_loss_epoch")
            frequency = lr_scheduler_kwargs.pop("frequency", 1)
            scheduler = self.lr_scheduler(
                optimizer_config["optimizer"], **lr_scheduler_kwargs
            )
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": monitor,
                "frequency": frequency,
            }

        return optimizer_config


class MLPAutoEncoder(BaseAutoEncoder):
    def __init__(
        self,
        # fixed params
        gene_dim: int,
        units_encoder: List[int],
        units_decoder: List[int],
        # params from datamodule
        batch_size: int,
        # model specific params
        supervised_subset: Optional[int] = None,
        reconstruction_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        dropout: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        output_activation: Callable[[], torch.nn.Module] = nn.Sigmoid,
        activation: Callable[[], torch.nn.Module] = nn.SELU,
        # params for masking
        masking_rate: Optional[float] = None,
        masking_strategy: Optional[str] = None,  # 'random', 'gene_program'
        encoded_gene_program: Optional[Dict] = None,
    ):
        # check input
        assert 0.0 <= dropout <= 1.0
        assert reconstruction_loss in ["mse", "mae", "continuous_bernoulli", "bce"]
        if reconstruction_loss in ["continuous_bernoulli", "bce"]:
            assert output_activation == nn.Sigmoid

        self.batch_size = batch_size
        self.supervised_subset = supervised_subset

        super(MLPAutoEncoder, self).__init__(
            gene_dim=gene_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

        self.encoder = MLP(
            in_channels=gene_dim,
            hidden_channels=units_encoder,
            activation_layer=activation,
            inplace=False,
            dropout=dropout,
        )
        # Define decoder network
        self.decoder = nn.Sequential(
            MLP(
                in_channels=units_encoder[-1],
                hidden_channels=units_decoder + [gene_dim],
                # norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
                activation_layer=activation,
                inplace=False,
                dropout=dropout,
            ),
            output_activation(),
        )

        self.predict_bottleneck = False

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

        # masking
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.encoded_gene_program = encoded_gene_program

        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.multiomics_indices = pickle.load(
            open(root + "/self_supervision/data/multiomics_indices.pickle", "rb")
        )

    def _step(self, batch, training=True):
        targets = batch["X"]
        inputs = batch["X"]

        if self.multiomics_indices is not None:
            inputs = inputs[:, self.multiomics_indices]
            targets = targets[:, self.multiomics_indices]
        if self.masking_rate and self.masking_strategy == "random":
            mask = (
                Bernoulli(probs=1.0 - self.masking_rate)
                .sample(targets.size())
                .to(targets.device)
            )
            # upscale inputs to compensate for masking and convert to same device
            masked_inputs = 1.0 / (1.0 - self.masking_rate) * (inputs * mask)
            x_latent, x_reconst = self(masked_inputs)
            # calculate masked loss on masked part only
            inv_mask = torch.abs(torch.ones(mask.size()).to(targets.device) - mask)
            loss = (
                inv_mask
                * self._calc_reconstruction_loss(x_reconst, targets, reduction="none")
            ).mean()

        elif self.masking_rate and self.masking_strategy == "gene_program":
            with torch.no_grad():
                # self.encoded gene program is a numpy array of encoded gene programs
                mask, frac = _mask_gene_programs_numpy(
                    inputs, self.encoded_gene_program, self.masking_rate
                )
                mask = torch.tensor(mask).to(inputs.device)
                # log the fraction of genes masked
                self.log("frac_genes_masked", frac)
                # mask, frac = self.mask_gene_programs(inputs, self.gene_program_dict, self.masking_rate)
            # upscale inputs to compensate for masking
            masked_inputs = 1.0 / (1.0 - frac) * (inputs * mask.to(inputs.device))
            x_latent, x_reconst = self(masked_inputs)
            # calculate masked loss
            inv_mask = torch.abs(torch.ones(mask.size()).to(targets.device) - mask)
            loss = (
                inv_mask.to(inputs.device)
                * self._calc_reconstruction_loss(x_reconst, targets, reduction="none")
            ).mean()

        elif self.masking_rate and self.masking_strategy == "single_gene_program":
            with torch.no_grad():
                # self.encoded gene program is a numpy array of encoded gene programs
                input_mask, output_mask = _mask_single_gene_programs(
                    inputs, self.encoded_gene_program
                )
                # log the fraction of genes masked
                self.log("frac_genes_masked", frac)
            # upscale inputs to compensate for masking
            masked_inputs = (
                1.0
                / (1.0 - frac)
                * (inputs * torch.tensor(input_mask).to(inputs.device))
            )
            x_latent, x_reconst = self(masked_inputs)
            # calculate masked loss only on the output_mask part of the reconstruction
            inv_output_mask = torch.abs(
                torch.ones(output_mask.size()).to(targets.device) - output_mask
            )
            loss = (
                torch.tensor(inv_output_mask).to(inputs.device)
                * self._calc_reconstruction_loss(x_reconst, targets, reduction="none")
            ).mean()

        elif self.masking_rate and self.masking_strategy == "gp_to_tf":
            with torch.no_grad():
                # self.encoded gene program is a Dict of encoded gene programs and the corresponding tf indices
                input_mask, output_mask = _only_activate_gene_program_numpy(
                    inputs, self.encoded_gene_program
                )
                # log the fraction of genes masked
                self.log("frac_genes_masked", frac)
            # upscale inputs to compensate for masking
            masked_inputs = (
                1.0
                / (1.0 - frac)
                * (inputs * torch.tensor(input_mask).to(inputs.device))
            )
            x_latent, x_reconst = self(masked_inputs)
            # calculate masked loss only on the output_mask part of the reconstruction
            inv_output_mask = torch.abs(
                torch.ones(output_mask.size()).to(targets.device) - output_mask
            )
            loss = (
                torch.tensor(inv_output_mask).to(inputs.device)
                * self._calc_reconstruction_loss(x_reconst, targets, reduction="none")
            ).mean()

        # raise error if masking rate is not none but masking strategy is not implemented
        elif self.masking_rate and self.masking_strategy not in [
            "random",
            "gene_program",
        ]:
            raise ValueError(
                f"Masking strategy {self.masking_strategy} not implemented."
            )

        # raise error if masking strategy is not none but masking rate is 0.0
        elif self.masking_strategy and self.masking_rate == 0.0:
            raise ValueError(
                f"Masking rate is 0 ({self.masking_rate}),"
                f" but masking strategy {self.masking_strategy} is not None."
            )

        else:
            x_latent, x_reconst = self(inputs)
            loss = self._calc_reconstruction_loss(x_reconst, targets, reduction="mean")

        return x_reconst, loss

    def predict_embedding(self, batch):
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        return self.encoder(batch["X"])

    def forward(self, x_in):
        x_latent = self.encoder(x_in)
        x_reconst = self.decoder(x_latent)
        return x_latent, x_reconst

    def training_step(self, batch, batch_idx):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}
        x_reconst, loss = self._step(batch)
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        self.log_dict(
            self.train_metrics(x_reconst, batch["X"]), on_epoch=True, on_step=True
        )
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}
        x_reconst, loss = self._step(batch, training=False)
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        self.log_dict(self.val_metrics(x_reconst, batch["X"]))
        self.log("val_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}
        x_reconst, loss = self._step(batch, training=False)
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        metrics = self.test_metrics(x_reconst, batch["X"])
        self.log_dict(metrics)
        self.log("test_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return metrics

    def predict_cell_types(self, x: torch.Tensor):
        return F.softmax(self(x)[0], dim=1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.supervised_subset is not None:
            mask = batch["dataset_id"] == self.supervised_subset
            if not any(mask):
                return  # Skip the batch if no items match the condition

            # Filter the batch based on the mask
            batch = {key: value[mask] for key, value in batch.items()}

        if batch_idx % self.gc_freq == 0:
            gc.collect()

        if hasattr(self, "predict_embedding") and self.predict_embedding:
            return self.encoder(batch["X"]).detach()

        else:
            x_reconst, _ = self._step(batch, training=False)
            return x_reconst

    def get_input(self, batch):
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        return batch["X"]


class VAE(MLPAutoEncoder):
    def __init__(
        self,
        units_encoder: List[int],
        vae_type: str = "simple_vae",  # Choose between 'simple_vae' and 'scvi_vae'
        **kwargs,
    ):
        super(VAE, self).__init__(units_encoder=units_encoder, **kwargs)

        self.vae_type = vae_type
        self.latent_dim = self.encoder[-1].out_features
        self.decoder[0].in_channels = self.latent_dim

        # Modify encoder to output both mean and log_var
        self.encoder_mu = nn.Linear(self.encoder[-1].out_features, self.latent_dim)
        self.encoder_log_var = nn.Linear(self.encoder[-1].out_features, self.latent_dim)

        # For scVI-style VAE
        if self.vae_type == "scvi_vae":
            self.encoder_pi = nn.Linear(
                self.latent_dim, self.encoder[0].in_features
            )  # from 64 to 2000
            self.encoder_theta = nn.Linear(
                self.latent_dim, self.encoder[0].in_features
            )  # from 64 to 2000

        # Modify decoder to take latent_dim as input
        self.decoder[0].in_channels = self.latent_dim

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, std) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def zinb_loss(self, y_true, y_pred, pi, theta):
        """
        Zero-inflated negative binomial loss.
        """
        nb_term = (
            torch.lgamma(theta + y_true)
            - torch.lgamma(y_true + 1)
            - torch.lgamma(theta)
        )
        nb_term += theta * (torch.log(theta) - torch.log(theta + y_pred)) + y_true * (
            torch.log(y_pred) - torch.log(theta + y_pred)
        )
        zero_term = torch.log(
            pi + (1 - pi) * torch.pow(theta / (theta + y_pred), theta)
        )
        return -torch.sum(zero_term + torch.logaddexp(torch.log(1 - pi), nb_term))

    def invert_sf_log1p_norm(self, x_norm):
        """
        Invert the log1p and scaling factor normalization
        """
        x_exp = torch.expm1(x_norm)  # inverse of log1p
        scaling_factor = 10000.0 / torch.sum(x_exp, axis=1, keepdim=True)
        x_raw = x_exp / scaling_factor
        return x_raw

    def forward(self, x_in):
        x_encoded = self.encoder(x_in)

        mu = self.encoder_mu(x_encoded)
        log_var = self.encoder_log_var(x_encoded)

        z = self.reparameterize(mu, log_var)

        if self.vae_type == "scvi_vae":
            pi = self.encoder_pi(x_encoded)
            theta = self.encoder_theta(x_encoded)
        else:
            pi, theta = None, None

        x_reconst = self.decoder(z)

        if self.vae_type == "scvi_vae":
            x_reconst_inverted = self.invert_sf_log1p_norm(x_reconst)
        else:
            x_reconst_inverted = x_reconst

        return z, x_reconst_inverted, mu, log_var, pi, theta

    def _step(self, batch, training=True):
        if self.multiomics_indices is not None:
            x_in = batch["X"][:, self.multiomics_indices]
        else:
            x_in = batch["X"]
        z, x_reconst_inverted, mu, log_var, pi, theta = self.forward(x_in)
        targets = x_in

        if self.vae_type == "simple_vae":
            reconst_loss = self._calc_reconstruction_loss(x_reconst_inverted, targets)
        else:  # scvi_vae
            reconst_loss = self.zinb_loss(targets, x_reconst_inverted, pi, theta)

        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        self.log("kl_divergence", kl_divergence, on_epoch=True)

        loss = reconst_loss + kl_divergence

        return x_reconst_inverted, loss


class MLPClassifier(BaseClassifier):
    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        class_weights: np.ndarray,
        child_matrix: np.ndarray,
        units: List[int],
        # params from datamodule
        batch_size: int,
        # model specific params
        supervised_subset: Optional[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
    ):
        super(MLPClassifier, self).__init__(
            gene_dim=gene_dim,
            type_dim=type_dim,
            class_weights=class_weights,
            child_matrix=child_matrix,
            batch_size=batch_size,
            supervised_subset=supervised_subset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

        self.classifier = MLP(
            in_channels=gene_dim,
            hidden_channels=units + [type_dim],
            dropout=dropout,
            inplace=False,
        )

    def _step(self, batch, training=True):
        logits = self(batch["X"])
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            preds_corrected, targets_corrected = self.hierarchy_correct(
                preds, batch["cell_type"]
            )
        if training:
            loss = F.cross_entropy(
                logits, batch["cell_type"], weight=self.class_weights
            )
        else:
            loss = F.cross_entropy(logits, targets_corrected)

        return preds_corrected, loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        if hasattr(self, "predict_embedding") and self.predict_embedding:
            return self.classifier[:12](
                batch["X"]
            ).detach()  # Get embeddings up to layer 12
        else:
            return F.softmax(self(batch["X"]), dim=1)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the embedding from the MLP.
        The computed embedding is the output before the last Linear layer of the MLP.
        """
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == len(self.classifier) - 2:
                break
        return x

    def predict_embedding(self, batch, batch_idx, dataloader_idx=None) -> torch.Tensor:
        """
        Function to predict the embedding from the given batch.
        """
        x = batch["X"]
        return self.forward_embedding(x)


class MLPBYOL(BaseAutoEncoder):
    def __init__(
        self,
        # fixed params
        gene_dim: int,
        units_encoder: List[int],
        # params from datamodule
        batch_size: int,
        # contrastive learning params
        backbone: str,  # MLP, TabNet
        augment_type: str,  # Gaussian, Uniform
        augment_intensity: float,
        use_momentum: bool,
        # model specific params
        lr: float = 0.005,
        weight_decay: float = 0.1,
        dropout: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        activation: Callable[[], torch.nn.Module] = nn.SELU,
    ):
        # check input
        assert 0.0 <= dropout <= 1.0

        self.batch_size = batch_size

        super(MLPBYOL, self).__init__(
            gene_dim=gene_dim,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

        # assign inner model, that will be trained using the BYOL / SimSiam framework
        self.backbone = backbone
        self.gene_dim = gene_dim
        self.units_encoder = units_encoder
        self.activation = activation
        self.dropout = dropout
        self.inner_model = self._get_inner_model()

        self.byol = BYOL(
            net=self.inner_model,
            image_size=self.gene_dim,
            augment_type=augment_type,
            augment_intensity=augment_intensity,
            batch_size=self.batch_size,
            use_momentum=use_momentum,
        )

        # This is so far only used for multiomics
        # Choose gene indices from NeurIPS dataset in the CellNet data

        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.multiomics_indices = pickle.load(
            open(root + "/self_supervision/data/multiomics_indices.pickle", "rb")
        )

    def _get_inner_model(self):
        if self.backbone == "MLP":
            self.inner_model = MLP(
                in_channels=self.gene_dim,
                hidden_channels=self.units_encoder,
                activation_layer=self.activation,
                inplace=False,
                dropout=self.dropout,
            )
        elif self.backbone == "TabNet":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return self.inner_model

    def _step(self, batch):
        loss = self.forward(batch)
        return loss

    def predict_embedding(self, batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return self.encoder(batch["X"])

    def forward(self, batch):
        if batch["X"].dim() == 3:
            batch["X"] = batch["X"].squeeze(1)
        if self.multiomics_indices is not None and batch["X"].shape[-1] == 19357:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        return self.byol(batch["X"])

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x_reconst, loss = self(batch)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return x_reconst


# helper
class DictAsAttributes:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


class MLPBarlowTwins(BaseAutoEncoder):
    def __init__(
        self,
        # fixed params
        gene_dim: int,
        # params from datamodule
        train_set_size: int,
        batch_size: int,
        # contrastive learning params
        CHECKPOINT_PATH: str,
        backbone: str,  # MLP, TabNet
        augment_intensity: float,
        units_encoder: List[int] = [256, 256, 40],
        learning_rate_weights: float = 0.2,
        learning_rate_biases: float = 0.0048,
        lambd: float = 0.0051,
        lr: float = 0.005,  # dummy here
        weight_decay: float = 1e-6,
        dropout: float = 0.0,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,  # dummy here
        lr_scheduler_kwargs: Dict = None,  # dummy here
        activation: Callable[[], torch.nn.Module] = nn.SELU,
        mode: str = "cellnet",  # cellnet or multiomics
    ):
        # check input
        assert 0.0 <= dropout <= 1.0

        super(MLPBarlowTwins, self).__init__(
            gene_dim=gene_dim,
            train_set_size=train_set_size,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            automatic_optimization=False,
        )

        self.save_hyperparameters(ignore=["gene_dim"])

        self.best_val_loss = np.inf
        self.best_train_loss = np.inf
        self.mode = mode
        self.train_set_size = train_set_size
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate_weights = learning_rate_weights
        self.learning_rate_biases = learning_rate_biases
        self.lambd = lambd

        self.loader_length = self.train_set_size // self.batch_size

        self.step = 0  # for optimizer

        # assign inner model, that will be trained using the BYOL / SimSiam framework
        self.backbone = backbone
        self.gene_dim = gene_dim
        self.units_encoder = units_encoder
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size

        self.transform = Transform(p=augment_intensity)
        self.CHECKPOINT_PATH = CHECKPOINT_PATH

        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.multiomics_indices = pickle.load(
            open(root + "/self_supervision/data/multiomics_indices.pickle", "rb")
        )

        # This is so far only used for multiomics
        # Choose gene indices from NeurIPS dataset in the CellNet data

        inner_model = self._get_inner_model()

        # Initialize the model-specific parameters
        self.dropout = dropout
        self.activation = activation
        self.units_encoder = units_encoder
        self.barlow_twins = self._get_barlow_twins_model(inner_model)

    def _get_barlow_twins_model(self, inner_model):
        args = (
            self._prepare_barlow_twins_args()
        )  # Prepare the Barlow Twins arguments (use the relevant params)
        self.barlow_twins = BarlowTwins(
            backbone=inner_model, args=args, mode=self.mode, dropout=self.dropout
        )
        return self.barlow_twins

    def _get_inner_model(self):
        if self.mode == "multiomics":
            n_genes = 2000
            n_proteins = 134
            n_batches = 12
            n_hidden = 256
            dropout = self.dropout
            n_latent = 40
            self.inner_model = nn.Sequential(
                nn.Linear(
                    in_features=n_genes + n_proteins + n_batches, out_features=n_hidden
                ),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=n_hidden + n_batches, out_features=n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=n_hidden + n_batches, out_features=n_latent),
            )

        elif self.backbone == "MLP":
            self.inner_model = MLP(
                in_channels=self.gene_dim,
                hidden_channels=self.units_encoder,
                activation_layer=self.activation,
                inplace=False,
                dropout=self.dropout,
            )
        else:
            raise NotImplementedError
        return self.inner_model

    def _prepare_barlow_twins_args(self):
        # Prepare Barlow Twins arguments based on the required parameters
        args = {
            # 'data': ...,  # Path to your dataset directory (from class B)
            # 'workers': 8,  # number of data loader workers
            "epochs": 1000,  # number of total epochs to run
            "batch_size": self.batch_size,  # mini-batch size
            "learning_rate_weights": self.learning_rate_weights,  # base learning rate for weights
            "learning_rate_biases": self.learning_rate_biases,
            # base learning rate for biases and batch norm parameters
            "weight-decay": self.weight_decay,  # weight decay
            "lambd": self.lambd,  # weight on off-diagonal terms
            "projector": "256-256-512-512",  # projector MLP
            # 'print_freq': 100,  # print frequency
            # 'checkpoint_dir': './checkpoint/'  # path to checkpoint directory
        }
        self.args = DictAsAttributes(args)
        return self.args

    def _step(self, batch):
        if self.mode == "multiomics":
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            gene = batch["X"].to(device)
            mask_all_protein = torch.zeros(gene.shape[0], 134).to(device)
            covariate = batch["batch"].to(device)
            x_in = torch.cat((gene, mask_all_protein, covariate), dim=1)
        else:
            # For Multiomics, batch['X'] has additional dim to squeeze
            if batch["X"].dim() == 3:
                batch["X"] = batch["X"].squeeze(1)
            if self.multiomics_indices is not None and batch["X"].shape[-1] == 19357:
                x_in = batch["X"][:, self.multiomics_indices]
            elif self.multiomics_indices is not None and batch["X"].shape[-1] == 19331:
                x_in = batch["X"][:, self.multiomics_indices]
            else:
                x_in = batch["X"]
        # Modify the forward pass to use the Barlow Twins model (from class B)
        loss = self(x_in)
        return loss

    def predict_embedding(self, batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return self.encoder(batch["X"])

    def forward(self, batch):
        # Apply two different augmentations to the same image
        y1, y2 = self.transform(batch)
        # Compute the Barlow Twins loss
        loss = self.barlow_twins(y1, y2)
        return loss

    def training_step(self, batch, batch_idx):
        # Increment the step counter
        self.step += 1
        # Initialize the optimizer
        param_weights = []
        param_biases = []
        for param in self.barlow_twins.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{"params": param_weights}, {"params": param_biases}]
        opt = LARS(
            parameters,
            lr=0,
            weight_decay=self.weight_decay,
            weight_decay_filter=True,
            lars_adaptation_filter=True,
        )
        # Adjust the learning rate
        adjust_learning_rate(self.args, opt, self.loader_length, self.step)
        # Zero the gradient before loss computation
        opt.zero_grad()
        # Calculate the loss
        loss = self._step(batch)
        # Manual backward pass and optimizer step
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, on_epoch=True)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        if loss < self.best_train_loss:
            self.best_train_loss = loss
            self.barlow_twins.save_model(
                path=self.CHECKPOINT_PATH + "/best_checkpoint_train.ckpt"
            )

        self.barlow_twins.save_model(
            path=self.CHECKPOINT_PATH + "/last_checkpoint.ckpt"
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.barlow_twins.save_model(
                path=self.CHECKPOINT_PATH + "/best_checkpoint_val.ckpt"
            )

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x_reconst, loss = self(batch)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return x_reconst
