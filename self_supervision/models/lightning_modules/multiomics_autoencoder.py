import lightning.pytorch as pl
from torch.distributions import Bernoulli, ContinuousBernoulli
from scvi.distributions import NegativeBinomial
from torchmetrics import MetricCollection
from typing import Optional, Dict, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from functools import partial
import os
import pickle
import gc
import abc
from torchmetrics import ExplainedVariance, MeanSquaredError
from self_supervision.models.contrastive.bt import Transform


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
        input_mask[ix, encoded_gene_program[key][1]] = 1
        output_mask[ix, encoded_gene_program[key][0]] = 1
    return input_mask, output_mask


"""### base class of autoencoder"""


class BaseAutoEncoder(pl.LightningModule, abc.ABC):
    autoencoder: nn.Module  # autoencoder mapping von n_genes to n_genes

    def __init__(
        self,
        # params from datamodule
        batch_size: int,
        # model specific params
        reconst_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        gc_frequency: int = 1,
        automatic_optimization: bool = True,
    ):
        super(BaseAutoEncoder, self).__init__()

        self.automatic_optimization = automatic_optimization

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


class MLPAutoEncoder(BaseAutoEncoder):
    def __init__(
        self,
        # fixed params
        # params from datamodule
        n_proteins: int = 134,
        batch_size: int = 8192,
        # multiomics specs
        n_genes=2000,
        n_batches=12,
        n_hidden=256,
        n_latent=40,
        n_data=90261,
        multiomics_type="atac",  # "cite"
        indices_file='multiomics_multi_indices',  # 'multiomics_cite_indices'
        n_projector_dim: Optional[int] = 512,
        gc_frequency: int = 5,
        # model specific params
        reconstruction_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        dropout: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        output_activation: Callable[[], torch.nn.Module] = nn.Sigmoid,
        # params for masking
        masking_rate: Optional[float] = 0.5,
        masking_strategy: Optional[str] = "random",  # 'random', 'gene_program'
        encoded_gene_program: Optional[
            Dict
        ] = None,  # only needed if masking_strategy == 'gene_program'
    ):
        # multiomics specs
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.n_batches = n_batches
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.n_projector_dim = n_projector_dim
        self.reconst_loss = "mse"
        self.dropout = dropout
        self.lr = learning_rate
        self.wd = weight_decay
        self.gc_freq = gc_frequency
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.encoded_gene_program = encoded_gene_program
        self.multiomics_type = multiomics_type

        # check input
        assert 0.0 <= dropout <= 1.0
        assert reconstruction_loss in ["mse", "mae", "continuous_bernoulli", "bce"]
        if reconstruction_loss in ["continuous_bernoulli", "bce"]:
            assert output_activation == nn.Sigmoid

        self.batch_size = batch_size
        self.dropout = dropout

        super(MLPAutoEncoder, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.save_hyperparameters(ignore=["n_genes", "n_proteins"])


        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(
                in_features=self.n_genes,
                out_features=self.n_hidden,
                bias=True,
            ),
            nn.BatchNorm1d(
                self.n_hidden,
                eps=0.001,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout, inplace=False),
            # Layer 2
            nn.Linear(
                in_features=self.n_hidden,
                out_features=self.n_hidden,
                bias=True,
            ),
            nn.BatchNorm1d(
                self.n_hidden,
                eps=0.001,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout, inplace=False),
            # Layer 3
            nn.Linear(
                in_features=self.n_hidden + self.n_batches,
                out_features=self.n_latent,
                bias=True,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.n_latent,
                out_features=self.n_hidden,
                bias=True,
            ),
            nn.BatchNorm1d(
                self.n_hidden,
                eps=0.001,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(
                in_features=self.n_hidden,
                out_features=self.n_hidden,
                bias=True,
            ),
            nn.BatchNorm1d(
                self.n_hidden,
                eps=0.001,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(
                in_features=self.n_hidden,
                out_features=self.n_genes,
                bias=True,
            ),
        )

        # masking
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.encoded_gene_program = encoded_gene_program

        # Choose gene indices from NeurIPS dataset in the CellNet data
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.multiomics_indices = pickle.load(
            open(root + "/self_supervision/data/" + indices_file + ".pickle", "rb")
        )

        self.train_iters_per_epoch = n_data // self.batch_size
        self.warmup_epochs = 10
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.encoded_gene_program = encoded_gene_program

    def _step(self, batch, training=True):
        targets = batch["X"]
        inputs = batch["X"]
        if inputs.dim() == 3:
            inputs = inputs.squeeze(1)

        if targets.dim() == 3:
            targets = targets.squeeze(1)

        if self.multiomics_type == "cite":
            proteins = torch.zeros((inputs.shape[0], 146))
            proteins = proteins.to(targets.device)

            if self.multiomics_indices is not None and inputs.shape[1] == 19331:
                inputs = inputs[:, self.multiomics_indices]
                targets = targets[:, self.multiomics_indices]

            # assert if inputs and targets shape is not batch size x 2000 (hvgs)
            assert (
                inputs.shape[1] == self.n_genes
            ), f"inputs shape is {inputs.shape} instead of {self.n_genes}"
            assert (
                targets.shape[1] == self.n_genes
            ), f"targets shape is {targets.shape} instead of {self.n_genes}"

            if self.masking_rate and self.masking_strategy == "random":
                mask = (
                    Bernoulli(probs=1.0 - self.masking_rate)
                    .sample(targets.size())
                    .to(targets.device)
                )
                # upscale inputs to compensate for masking and convert to same device
                masked_inputs = 1.0 / (1.0 - self.masking_rate) * (inputs * (1 - mask))
                masked_inputs = torch.cat((masked_inputs, proteins), dim=1)
                targets = torch.cat((targets, proteins), dim=1)
                x_latent, x_reconst = self(masked_inputs)
                # calculate masked loss
                loss = (
                    mask
                    * self._calc_reconstruction_loss(
                        x_reconst[:, : self.n_genes],
                        targets[:, : self.n_genes],
                        reduction="none",
                    )
                ).mean()

            elif self.masking_rate and self.masking_strategy == "gene_program":
                with torch.no_grad():
                    mask, frac = _mask_gene_programs_numpy(
                        inputs=inputs,
                        encoded_gene_program=self.encoded_gene_program,
                        masking_rate=self.masking_rate,
                    )
                    # log the fraction of genes masked
                    self.log("frac_genes_masked", frac)
                    # mask, frac = self.mask_gene_programs(inputs, self.gene_program_dict, self.masking_rate)
                # upscale inputs to compensate for masking
                masked_inputs = (
                    1.0 / (1.0 - frac) * (inputs * torch.tensor(1 - mask).to(inputs.device))
                )
                proteins = proteins.to(targets.device)
                masked_inputs = torch.cat((masked_inputs, proteins), dim=1)
                targets = torch.cat((targets, proteins), dim=1)
                x_latent, x_reconst = self(masked_inputs)
                # calculate masked loss
                loss = (
                    torch.tensor(mask).to(inputs.device)
                    * self._calc_reconstruction_loss(
                        x_reconst[:, : self.n_genes],
                        targets[:, : self.n_genes],
                        reduction="none",
                    )
                ).mean()

            elif self.masking_rate and self.masking_strategy == "gp_to_tf":
                with torch.no_grad():
                    # self.encoded gene program is a Dict of encoded gene programs and the corresponding tf indices
                    input_mask, output_mask = _only_activate_gene_program_numpy(
                        inputs, self.encoded_gene_program
                    )
                    input_mask = torch.tensor(input_mask).to(inputs.device)
                    output_mask = torch.tensor(output_mask).to(inputs.device)
                    frac = torch.sum(input_mask).item() / (
                        input_mask.shape[0] * input_mask.shape[1]
                    )
                    # log the fraction of genes masked
                    self.log("frac_genes_masked", frac)
                # upscale inputs to compensate for masking
                masked_inputs = 1.0 / (1.0 - frac) * (inputs * input_mask.to(inputs.device))
                proteins = proteins.to(targets.device)
                masked_inputs = torch.cat((masked_inputs, proteins), dim=1)
                targets = torch.cat((targets, proteins), dim=1)
                x_latent, x_reconst = self(masked_inputs)
                output_mask = torch.cat((output_mask, proteins), dim=1)
                # calculate masked loss only on the output_mask part of the reconstruction
                loss = (
                    torch.tensor(output_mask[:, : self.n_genes]).to(inputs.device)
                    * self._calc_reconstruction_loss(
                        x_reconst[:, : self.n_genes],
                        targets[:, : self.n_genes],
                        reduction="none",
                    )
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
        
        elif self.multiomics_type == "atac":
            # autoencoder pretraining from rna <-> rna
            # in finetuning from rna+batch -> atac
            # no proteins, input is only rna + batch
            # print('Debug print for ATAC')
            if self.multiomics_indices is not None and inputs.shape[1] == 19331:
                # scTab data -> subset to multiomics genes
                inputs = inputs[:, self.multiomics_indices]
                targets = targets[:, self.multiomics_indices]
                # NeurIPS data already subsetted


            batch = torch.zeros((inputs.shape[0], self.n_batches)).to(inputs.device)
            inputs = torch.cat((inputs, batch), dim=1)

            # assert if inputs and targets shape is not batch size x 2000 (hvgs)
            if self.masking_rate and self.masking_strategy == "random":
                # print('Random masking for ATAC')
                mask = (
                    Bernoulli(probs=1.0 - self.masking_rate)
                    .sample(targets.size())
                    .to(targets.device)
                )
                mask_in = torch.cat((mask, batch), dim=1)   
                # upscale inputs to compensate for masking and convert to same device
                masked_inputs = 1.0 / (1.0 - self.masking_rate) * (inputs * (1 - mask_in))
                x_latent, x_reconst = self(masked_inputs)
                # calculate masked loss
                # print('Before loss shapes: ', x_reconst.shape, targets.shape, mask.shape)
                loss = (
                    mask
                    * self._calc_reconstruction_loss(
                        x_reconst,
                        targets,
                        reduction="none",
                    )
                ).mean()

            else:
                raise ValueError("For now, only random masking is supported for ATAC")
            
            return x_reconst, loss
    


    def predict_embedding(self, batch):
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        return self.encoder(batch["X"])

    def forward(self, x_in):
        if self.multiomics_type == "cite":
            covariate = x_in[:, self.n_genes + self.n_proteins :]
        else:
            covariate = x_in[:, self.n_genes :]
        x = x_in[:, : self.n_genes] 
        for i in range(0, len(self.encoder)):
            if (i % 7 == 0) & (i != 0):
                x = torch.cat((x, covariate), dim=1)
                x = self.encoder[i](x)
            else:
                x = self.encoder[i](x)
        x_latent = x
        for i in range(0, len(self.decoder)):

            x = self.decoder[i](x)
        x_reconst = x
        return x_latent, x_reconst

    def training_step(self, batch, batch_idx):
        x_reconst, loss = self._step(batch)
        # to do add hvg here!
        if self.multiomics_indices is not None and batch["X"].shape[1] == 19331:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        # self.log_dict(self.train_metrics(x_reconst, batch['X']), on_epoch=True, on_step=True)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        x_reconst, loss = self._step(batch, training=False)
        if self.multiomics_indices is not None and batch["X"].shape[1] == 19331:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        # self.log_dict(self.val_metrics(x_reconst, batch['X']))
        self.log("val_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        x_reconst, loss = self._step(batch, training=False)
        if self.multiomics_indices is not None and batch["X"].shape[1] == 19331:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        if batch["X"].dim() == 3:
            batch["X"] = batch["X"].squeeze(1)
        metrics = self.test_metrics(x_reconst, batch["X"])
        self.log_dict(metrics)
        self.log("test_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return metrics

    def predict_cell_types(self, x: torch.Tensor):
        return F.softmax(self(x)[0], dim=1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x_reconst, loss = self._step(batch, training=False)
        x_true = batch["X"]
        if self.multiomics_indices is not None:
            x_true = x_true[:, self.multiomics_indices]
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return x_true, x_reconst

    def get_input(self, batch):
        if self.multiomics_indices is not None:
            batch["X"] = batch["X"][:, self.multiomics_indices]
        return batch["X"]


"""### main class of negative binomial autoencoder"""


"""### main class of masked autoencoder"""

class MultiomicsMultiAutoencoder(pl.LightningModule):
    def __init__(
        self,
        mode,  # 'pre_training','fine_tuning','no_mask'
        model,  # 'MAE','BT','BYOL'
        n_genes=2000,
        n_atac=116490,
        # multiomics specs
        n_batches=12,
        n_hidden=256,
        n_latent=40,
        batch_size=256,
        n_data=90261,
        masking_rate: Optional[float] = 0.5,
        masking_strategy: Optional[str] = None,  # 'random', 'gene_program','gp_to_tf'
        encoded_gene_program: Optional[list] = None,
        learning_rate=1e-05,
        dropout=0.2,
        weight_decay=1e-04,
        gc_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_genes = n_genes
        self.n_atac = n_atac
        self.n_batches = n_batches
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.reconst_loss = "mse"
        self.dropout = dropout
        self.lr = learning_rate
        self.wd = weight_decay
        self.mode = mode
        self.model = model
        self.gc_freq = gc_frequency
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(
                in_features=self.n_genes,
                out_features=self.n_hidden,
                bias=True,
            ),
            nn.BatchNorm1d(
                self.n_hidden,
                eps=0.001,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout, inplace=False),
            # Layer 2
            nn.Linear(
                in_features=self.n_hidden,
                out_features=self.n_hidden,
                bias=True,
            ),
            nn.BatchNorm1d(
                self.n_hidden,
                eps=0.001,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout, inplace=False),
            # Layer 3
            nn.Linear(
                in_features=self.n_hidden + self.n_batches,
                out_features=self.n_latent,
                bias=True,
            ),
        )

        rna_decoder = nn.Sequential(
            nn.Linear(
                in_features=self.n_latent, 
                out_features=self.n_hidden
            ),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.n_hidden, 
                out_features=self.n_hidden
            ),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.n_hidden,
                out_features=self.n_genes
            ),
        )


        atac_decoder = nn.Sequential(
            nn.Linear(
                in_features=self.n_latent, out_features=4*self.n_hidden
            ),  
            nn.BatchNorm1d(4*self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=4*self.n_hidden, out_features=32*self.n_hidden
            ),
            nn.BatchNorm1d(32*self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=32*self.n_hidden,
                out_features=self.n_atac,
            ),
        )
        
        if mode == 'pre_training':  # RNA-RNA
            self.decoder = rna_decoder
        elif mode == 'fine_tuning':  # RNA-ATAC
            self.decoder = atac_decoder
        else:
            raise ValueError("Unsupported mode: " + mode)

        # Choose gene indices from NeurIPS dataset in the CellNet data
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.multiomics_indices = pickle.load(
            open(root + "/self_supervision/data/multiomics_multi_indices.pickle", "rb")
        )

        self.train_iters_per_epoch = n_data // self.batch_size
        self.warmup_epochs = 10
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.encoded_gene_program = encoded_gene_program

    def forward(self, x):
        x = x.to(self.device)
        covariate = x[:, self.n_genes:]
        x = x[:, : self.n_genes]
        
        for i in range(len(self.encoder)):
            if i % 7 == 0 and i != 0:
                x = torch.cat((x, covariate), dim=1)

            x = self.encoder[i](x)
        
        x_latent = x
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        
        x_reconst = x
        return x_latent, x_reconst


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

    def _step(self, batch, training=True):        
        # Handle different types of batches
        if isinstance(batch, tuple):  
            gene = batch[0]["X"]
            atac = batch[0]["atac"]
        elif isinstance(batch, dict):
            gene = batch["X"]
            atac = batch["atac"]
        else:
            raise ValueError("Batch must be a dictionary with keys 'X', 'atac', and 'batch'.")

        # Convert gene to a torch tensor if it's a numpy array
        if isinstance(gene, np.ndarray):
            gene = torch.tensor(gene).float()

        # Ensure gene has the correct dimensions
        if gene.dim() == 3:
            gene = gene.squeeze(1)
        elif gene.dim() == 1:
            gene = gene.unsqueeze(0)

        # Handle covariate based on the mode and presence in the batch
        if "covariate" in batch and self.mode == "fine_tuning":
            covariate = batch["covariate"]
        else:
            covariate = torch.zeros(gene.shape[0], self.n_batches).to(gene.device)

        # Adjust gene shape based on dataset type
        if gene.shape[-1] == 2000:  # NeurIPS data
            pass
        elif gene.shape[-1] == 19331:  # CellNet data
            assert max(self.multiomics_indices) < gene.shape[1], "Index out of range"
            assert min(self.multiomics_indices) >= 0, "Index out of range"
            gene = gene[:, self.multiomics_indices]
        else:
            raise ValueError("Unsupported gene shape: " + str(gene.shape))
        
        if self.mode == "fine_tuning":
            inputs = torch.cat((gene, covariate), dim=1)
            x_latent, x_reconst = self.forward(inputs)
            return x_latent, x_reconst, self._calc_reconstruction_loss(x_reconst, atac, reduction="none").mean()

        if self.mode != "pre_training":
            raise ValueError(f"Invalid mode {self.mode}. Choose either pre_training or fine_tuning.")

        if self.masking_strategy == "random":
            mask = Bernoulli(probs=1.0 - self.masking_rate).sample(gene.size()).to(gene.device)
            assert torch.all((mask == 0) | (mask == 1)), "mask contains values other than 0 or 1"

            masked_genes = 1.0 / (1.0 - self.masking_rate) * (gene * mask)
            masked_inputs = torch.cat((masked_genes, covariate), dim=1)
            x_latent, x_reconst = self(masked_inputs)

            inv_mask = torch.abs(torch.ones(mask.size()).to(gene.device) - mask)
            # print('Calculate loss between x_reconst, ', x_reconst.shape, ' and gene, ', gene.shape)
            return x_latent, x_reconst, (inv_mask * self._calc_reconstruction_loss(x_reconst, gene, reduction="none")).mean()

        if self.masking_strategy == "gene_program":
            with torch.no_grad():
                mask, masking_rate = _mask_gene_programs_numpy(
                    inputs=gene,
                    encoded_gene_program=self.encoded_gene_program,
                    masking_rate=self.masking_rate,
                )
                mask = torch.tensor(mask).to(gene.device)
                self.log("frac_genes_masked", masking_rate.astype(np.float32))
                assert torch.all((mask == 0) | (mask == 1)), "mask contains values other than 0 or 1"

            masked_genes = 1.0 / (1.0 - self.masking_rate) * (gene * mask)
            masked_inputs = torch.cat((masked_genes, covariate), dim=1)
            x_latent, x_reconst = self.forward(masked_inputs)

            inv_mask = torch.abs(torch.ones(mask.size()).to(gene.device) - mask)
            return x_latent, x_reconst, (inv_mask * self._calc_reconstruction_loss(x_reconst[:, :self.n_genes], gene, reduction="none")).mean()

        if self.masking_strategy in ["gp_to_tf", "gp_to_gp"]:
            with torch.no_grad():
                input_mask, output_mask = _only_activate_gene_program_numpy(gene, self.encoded_gene_program)
                input_mask = torch.tensor(input_mask).to(gene.device)
                frac = torch.sum(input_mask).item() / (input_mask.shape[0] * input_mask.shape[1])
                output_mask = torch.tensor(output_mask).to(gene.device)
                # self.log('frac_genes_masked', frac)

            masked_genes = 1.0 / (1.0 - frac) * (gene * (1 - input_mask))
            masked_inputs = torch.cat((masked_genes, covariate), dim=1)
            x_latent, x_reconst = self.forward(masked_inputs)

            loss_mask = input_mask if self.masking_strategy == "gp_to_gp" else output_mask
            return x_latent, x_reconst, (loss_mask * self._calc_reconstruction_loss(x_reconst[:, :self.n_genes], gene, reduction="none")).mean()

        raise ValueError(f"Invalid masking strategy {self.masking_strategy}. Choose random, gene_program, gp_to_tf, or gp_to_gp.")

    def _fn(self, warmup_steps, step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 1.0

    def _linear_warmup_decay(self, warmup_steps):
        return partial(self._fn, warmup_steps)

    def training_step(self, batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        _, _, loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        _, _, loss = self._step(val_batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        x_latent, x_reconst, loss = self._step(batch)
        self.log("test_loss", loss)
        return x_latent, x_reconst

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        x_latent, x_reconst, loss = self._step(batch)
        return x_latent, x_reconst

    def get_latent_embedding(self, batch):
        x_latent, _, _ = self._step(batch)
        return x_latent.detach().numpy()

    def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity loss between x and y.
        :param x: Input tensor of shape (batch_size, embedding_dim)
        :param y: Input tensor of shape (batch_size, embedding_dim)
        :return: Tensor of shape (batch_size,) representing the cosine similarity loss between x and y.
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def configure_optimizers(self):
        if self.model == "BT":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

            warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    self._linear_warmup_decay(warmup_steps),
                ),
                "interval": "step",
                "frequency": 1,
            }

            return [optimizer], [scheduler]
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
            return optimizer
    
    
class MultiomicsCiteAutoencoder(pl.LightningModule):
    def __init__(
        self,
        mode,  # 'pre_training','fine_tuning','no_mask'
        model,  # 'MAE','BT','BYOL'
        n_genes=2000,
        n_proteins=134,
        n_batches=12,
        n_hidden=256,
        n_latent=40,
        batch_size=256,
        n_data=90261,
        intensity: Optional[float] = 0.0001,
        lambda_coeff: Optional[float] = 0.0051,
        masking_rate: Optional[float] = 0.5,
        masking_strategy: Optional[str] = None,  # 'random', 'gene_program','gp_to_tf'
        encoded_gene_program: Optional[list] = None,
        learning_rate=1e-05,
        dropout=0.2,
        weight_decay=1e-04,
        gc_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.n_batches = n_batches
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.reconst_loss = "mse"
        self.dropout = dropout
        self.lr = learning_rate
        self.wd = weight_decay
        self.mode = mode
        self.model = model
        self.lambda_coeff = lambda_coeff
        self.gc_freq = gc_frequency
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=self.n_genes + self.n_proteins + self.n_batches,
                out_features=self.n_hidden,
            ),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.n_hidden + self.n_batches, out_features=self.n_hidden
            ),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.n_hidden + self.n_batches, out_features=self.n_latent
            ),
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.n_latent + self.n_batches, out_features=self.n_hidden
            ),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.n_hidden + self.n_batches, out_features=self.n_hidden
            ),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=self.n_hidden + n_batches,
                out_features=self.n_genes + self.n_proteins + self.n_batches,
            ),
        )

        # Choose gene indices from NeurIPS dataset in the CellNet data
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.multiomics_indices = pickle.load(
            open(root + "/self_supervision/data/multiomics_cite_indices.pickle", "rb")
        )

        self.train_iters_per_epoch = n_data // self.batch_size
        self.warmup_epochs = 10
        self.transform = Transform(negbin_intensity=intensity, dropout_intensity=intensity)
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy
        self.encoded_gene_program = encoded_gene_program

    def forward(self, x):
        x = x.to(self.device)
        covariate = x[:, self.n_genes + self.n_proteins :]
        for i in range(0, len(self.encoder)):
            if (i % 4 == 0) & (i != 0):
                x = torch.cat((x, covariate), dim=1)
                x = self.encoder[i](x)
            else:
                x = self.encoder[i](x)
        x_latent = x
        for i in range(0, len(self.decoder)):
            if i % 4 == 0:
                x = torch.cat((x, covariate), dim=1)
                x = self.decoder[i](x)
            else:
                x = self.decoder[i](x)
        x_reconst = x
        return x_latent, x_reconst

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

    def _step(self, batch, training=True):
        # if protein is not a key of batch (in RNA-seq data), initialize protein and batch as zeros
        if isinstance(batch, tuple):  # Pretraining - mask protein
            gene = batch[0]["X"]
            protein = torch.zeros(gene.shape[0], 134).to(gene.device)
            covariate = torch.zeros(gene.shape[0], 12).to(gene.device)
        elif isinstance(batch, dict):
            if "protein" not in batch.keys():  # Pretraining - mask protein
                gene = batch["X"]
                protein = torch.zeros(gene.shape[0], 134).to(gene.device)
                covariate = torch.zeros(gene.shape[0], 12).to(gene.device)
            else:  # Fine-tuning - don't mask protein
                gene = batch["X"]
                protein = batch["protein"]
                covariate = batch["batch"]
        else:
            raise ValueError("Unsupported batch type: " + str(type(batch)))

        if gene.dim() == 3:
            gene = gene.squeeze(1)

        if gene.shape[-1] == 2000:  # NeurIPS data
            # Don't change shape
            pass
        elif gene.shape[-1] == 19331:  # CellNet data
            # Check Ranges of Indices: Ensure all values within the list self.multiomics_indices are within valid range i.e., [0, gene.shape[1]-1].
            # assert self.multiomics_indices.max().item() < gene.shape[1], "Index out of range"
            # assert self.multiomics_indices.min().item() >= 0, "Index out of range"

            assert max(self.multiomics_indices) < gene.shape[1], "Index out of range"
            assert min(self.multiomics_indices) >= 0, "Index out of range"

            torch.cuda.synchronize()
            gene = gene[:, self.multiomics_indices]
            torch.cuda.synchronize()
        else:
            raise ValueError("Unsupported gene shape: " + str(gene.shape))

        mask_all_protein = torch.zeros_like(protein)

        if (
            self.model == "MAE"
            and self.mode == "pre_training"
            and self.masking_strategy == "random"
        ):
            mask = (
                Bernoulli(probs=1.0 - self.masking_rate)
                .sample(gene.size())
                .to(gene.device)
            )
            assert torch.all(
                (mask == 0) | (mask == 1)
            ), "mask contains values other than 0 or 1"

            # upscale inputs to compensate for masking and convert to same device
            # masked_genes = 1. / (1. - self.masking_rate) * (gene * (1-mask))
            masked_genes = (
                1.0 / (1.0 - self.masking_rate) * (gene * mask)
            )  # batch_size x 2000
            masked_inputs = torch.cat(
                (masked_genes, mask_all_protein, covariate), dim=1
            )  # batch_size x 2146
            x_latent, x_reconst = self(masked_inputs)  # x_reconst batch_size x 2146

            inv_mask = torch.abs(torch.ones(mask.size()).to(gene.device) - mask)
            # calculate masked loss
            loss = (
                inv_mask
                * self._calc_reconstruction_loss(
                    x_reconst[:, : self.n_genes], gene, reduction="none"
                )
            ).mean()

        elif (
            self.model == "MAE"
            and self.mode == "pre_training"
            and self.masking_strategy == "gene_program"
        ):
            with torch.no_grad():
                mask, masking_rate = _mask_gene_programs_numpy(
                    inputs=gene,
                    encoded_gene_program=self.encoded_gene_program,
                    masking_rate=self.masking_rate,
                )
                mask = torch.tensor(mask).to(gene.device)
                # log the fraction of genes masked
                self.log("frac_genes_masked", masking_rate.astype(np.float32))
                assert torch.all(
                    (mask == 0) | (mask == 1)
                ), "mask contains values other than 0 or 1"

            # upscale inputs to compensate for masking
            # masked_genes = 1. / (1. - masking_rate) * (gene * (1-mask))
            # get number of ones per cell in mask
            masked_genes = 1.0 / (1.0 - self.masking_rate) * (gene * mask)
            masked_inputs = torch.cat(
                (masked_genes, mask_all_protein, covariate), dim=1
            )
            x_latent, x_reconst = self.forward(masked_inputs)
            # calculate masked loss
            inv_mask = torch.abs(torch.ones(mask.size()).to(gene.device) - mask)
            loss = (
                inv_mask
                * self._calc_reconstruction_loss(
                    x_reconst[:, : self.n_genes], gene, reduction="none"
                )
            ).mean()
        elif (
            self.model == "MAE"
            and self.mode == "pre_training"
            and self.masking_strategy == "gp_to_tf"
        ):
            with torch.no_grad():
                # self.encoded gene program is a Dict of encoded gene programs and the corresponding tf indices
                input_mask, output_mask = _only_activate_gene_program_numpy(
                    gene, self.encoded_gene_program
                )
                input_mask = torch.tensor(input_mask).to(gene.device)
                frac = torch.sum(input_mask).item() / (
                    input_mask.shape[0] * input_mask.shape[1]
                )
                output_mask = torch.tensor(output_mask).to(gene.device)
                # log the fraction of genes masked
                # self.log('frac_genes_masked', frac)
            # upscale inputs to compensate for masking
            masked_genes = 1.0 / (1.0 - frac) * (gene * (1 - input_mask))
            masked_genes = masked_genes
            masked_inputs = torch.cat(
                (masked_genes, mask_all_protein, covariate), dim=1
            )
            x_latent, x_reconst = self.forward(masked_inputs)
            # calculate masked loss
            loss = (
                output_mask
                * self._calc_reconstruction_loss(
                    x_reconst[:, : self.n_genes], gene, reduction="none"
                )
            ).mean()
        elif (
            self.model == "MAE"
            and self.mode == "pre_training"
            and self.masking_strategy == "gp_to_gp"
        ):
            with torch.no_grad():
                # self.encoded gene program is a Dict of encoded gene programs and the corresponding tf indices
                input_mask, output_mask = _only_activate_gene_program_numpy(
                    gene, self.encoded_gene_program
                )
                input_mask = torch.tensor(input_mask).to(gene.device)
                frac = torch.sum(input_mask).item() / (
                    input_mask.shape[0] * input_mask.shape[1]
                )
                output_mask = torch.tensor(output_mask).to(gene.device)
                # log the fraction of genes masked
                # self.log('frac_genes_masked', frac)
            # upscale inputs to compensate for masking
            masked_genes = 1.0 / (1.0 - frac) * (gene * (1 - input_mask))
            masked_genes = masked_genes
            masked_inputs = torch.cat(
                (masked_genes, mask_all_protein, covariate), dim=1
            )
            x_latent, x_reconst = self.forward(masked_inputs)
            # calculate masked loss
            loss = (
                input_mask
                * self._calc_reconstruction_loss(
                    x_reconst[:, : self.n_genes], gene, reduction="none"
                )
            ).mean()

        elif self.mode == "fine_tuning":
            inputs = torch.cat((gene, mask_all_protein, covariate), dim=1)
            x_latent, x_reconst = self.forward(inputs)
            loss = self._calc_reconstruction_loss(
                x_reconst[:, self.n_genes : self.n_genes + self.n_proteins],
                protein,
                reduction="none",
            ).mean()
        else:
            raise ValueError(
                "Please choose either full_run or fine-tuning or a masking strategy for pre-training. "
                "You chose " + self.mode + " and " + self.masking_strategy
            )

        return x_latent, x_reconst, loss

    def _fn(self, warmup_steps, step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 1.0

    def _linear_warmup_decay(self, warmup_steps):
        return partial(self._fn, warmup_steps)

    def training_step(self, batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        _, _, loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        _, _, loss = self._step(val_batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        x_latent, x_reconst, loss = self._step(batch)
        self.log("test_loss", loss)
        return x_latent, x_reconst

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        x_latent, x_reconst, loss = self._step(batch)
        return x_latent, x_reconst[:, self.n_genes : self.n_genes + self.n_proteins]

    def get_latent_embedding(self, batch):
        x_latent, _, _ = self._step(batch)
        return x_latent.detach().numpy()

    def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity loss between x and y.
        :param x: Input tensor of shape (batch_size, embedding_dim)
        :param y: Input tensor of shape (batch_size, embedding_dim)
        :return: Tensor of shape (batch_size,) representing the cosine similarity loss between x and y.
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def configure_optimizers(self):
        if self.model == "BT":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

            warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    self._linear_warmup_decay(warmup_steps),
                ),
                "interval": "step",
                "frequency": 1,
            }

            return [optimizer], [scheduler]
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
            return optimizer
