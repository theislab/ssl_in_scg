# Adapted from:
# https://github.com/theislab/cellnet/blob/main/cellnet/estimators.py
from os.path import join
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from torch.utils.data import Sampler

from self_supervision.data.datamodules import MerlinDataModule
from self_supervision.models.lightning_modules.cellnet_autoencoder import (
    MLPAutoEncoder,
    MLPClassifier,
    MLPBYOL,
    MLPBarlowTwins,
    VAE,
    MLPNegBin,
    VAENegBin,
)


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class EstimatorAutoEncoder:
    datamodule: MerlinDataModule
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(self, data_path: str, hvg: bool = False, num_hvgs: int = 2000):
        self.data_path = data_path
        self.hvg = hvg
        self.num_hvgs = num_hvgs

    def init_datamodule(
        self,
        batch_size: int = 2048,
        sub_sample_frac: float = 1.0,
        dataloader_kwargs_train: Dict = None,
        dataloader_kwargs_inference: Dict = None,
        merlin_dataset_kwargs_train: Dict = None,
        merlin_dataset_kwargs_inference: Dict = None,
    ):
        self.datamodule = MerlinDataModule(
            self.data_path,
            columns=["cell_type", "dataset_id"],
            batch_size=batch_size,
            sub_sample_frac=sub_sample_frac,
            dataloader_kwargs_train=dataloader_kwargs_train,
            dataloader_kwargs_inference=dataloader_kwargs_inference,
            dataset_kwargs_train=merlin_dataset_kwargs_train,
            dataset_kwargs_inference=merlin_dataset_kwargs_inference,
        )

    def init_model(self, model_type: str, model_kwargs):
        if model_type == "mlp_ae":
            self.model = MLPAutoEncoder(
                **{**self.get_fixed_autoencoder_params(), **model_kwargs}
            )
        elif model_type == "mlp_vae":
            self.model = VAE(**{**self.get_fixed_autoencoder_params(), **model_kwargs})
        elif model_type == "mlp_negbin":
            self.model = MLPNegBin(
                **{**self.get_fixed_autoencoder_params(), **model_kwargs}
            )
        elif model_type == "mlp_negbin_vae":
            self.model = VAENegBin(
                **{**self.get_fixed_autoencoder_params(), **model_kwargs}
            )
        elif model_type == "mlp_byol":  # Bootstrap Your Own Latent
            self.model = MLPBYOL(
                **{**self.get_fixed_autoencoder_params(), **model_kwargs}
            )
        elif model_type == "mlp_bt":  # BarlowTwins
            self.model = MLPBarlowTwins(
                **{**self.get_fixed_autoencoder_params(), **model_kwargs}
            )
        elif model_type == "mlp_clf":
            self.model = MLPClassifier(
                **{**self.get_fixed_clf_params(), **model_kwargs}
            )
        else:
            raise ValueError(
                f'model_type has to be in ["mlp_ae, mlp_byol, mlp_clf"]. You supplied: {model_type}'
            )

    def init_trainer(self, trainer_kwargs):
        self.trainer = pl.Trainer(**trainer_kwargs)

    def _check_is_initialized(self):
        if not self.model:
            raise RuntimeError(
                "You need to call self.init_model before calling self.train"
            )
        if not self.datamodule:
            raise RuntimeError(
                "You need to call self.init_datamodule before calling self.train"
            )
        if not self.trainer:
            raise RuntimeError(
                "You need to call self.init_trainer before calling self.train"
            )

    def get_fixed_autoencoder_params(self):
        if self.hvg:
            return {
                "gene_dim": self.num_hvgs,
                "train_set_size": sum(self.datamodule.train_dataset.partition_lens),
                "val_set_size": sum(self.datamodule.val_dataset.partition_lens),
                "batch_size": self.datamodule.batch_size,
                "hvg": self.hvg,
                "num_hvgs": self.num_hvgs,
            }
        else:
            return {
                "gene_dim": len(pd.read_parquet(join(self.data_path, "var.parquet"))),
                "train_set_size": sum(self.datamodule.train_dataset.partition_lens),
                "val_set_size": sum(self.datamodule.val_dataset.partition_lens),
                "batch_size": self.datamodule.batch_size,
                "hvg": self.hvg,
                "num_hvgs": self.num_hvgs,
            }

    def get_fixed_clf_params(self):
        if self.hvg:
            return {
                "gene_dim": self.num_hvgs,
                "class_weights": np.load(join(self.data_path, "class_weights.npy")),
                "type_dim": len(
                    pd.read_parquet(
                        join(self.data_path, "categorical_lookup/cell_type.parquet")
                    )
                ),
                "child_matrix": np.load(
                    join(self.data_path, "cell_type_hierarchy/child_matrix.npy")
                ),
                "train_set_size": sum(self.datamodule.train_dataset.partition_lens),
                "val_set_size": sum(self.datamodule.val_dataset.partition_lens),
                "batch_size": self.datamodule.batch_size,
                "hvg": self.hvg,
                "num_hvgs": self.num_hvgs,
            }
        else:
            return {
                "gene_dim": len(pd.read_parquet(join(self.data_path, "var.parquet"))),
                "type_dim": len(
                    pd.read_parquet(
                        join(self.data_path, "categorical_lookup/cell_type.parquet")
                    )
                ),
                "class_weights": np.load(join(self.data_path, "class_weights.npy")),
                "child_matrix": np.load(
                    join(self.data_path, "cell_type_hierarchy/child_matrix.npy")
                ),
                "train_set_size": sum(self.datamodule.train_dataset.partition_lens),
                "val_set_size": sum(self.datamodule.val_dataset.partition_lens),
                "batch_size": self.datamodule.batch_size,
                "hvg": self.hvg,
                "num_hvgs": self.num_hvgs,
            }

    def find_lr(self, lr_find_kwargs, plot_results: bool = False):
        self._check_is_initialized()
        lr_finder = self.trainer.tuner.lr_find(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            **lr_find_kwargs,
        )
        if plot_results:
            lr_finder.plot(suggest=True)

        return lr_finder.suggestion(), lr_finder.results

    def train(self, ckpt_path: str = None):
        self._check_is_initialized()
        self.trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=ckpt_path,
        )

    def validate(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.validate(
            self.model,
            dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=ckpt_path,
        )

    def test(
        self,
        ckpt_path: str = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self._check_is_initialized()
        if dataloader is None:
            dataloader = self.datamodule.test_dataloader()
        return self.trainer.test(
            self.model, dataloaders=dataloader, ckpt_path=ckpt_path
        )

    def predict(
        self, dataloader=None, ckpt_path: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_initialized()
        # This assumes that trainer.predict returns a tuple of two lists of tensors,
        # the first being the predictions and the second being the true values

        # For debugging, sometimes only one value is returned, print that value
        out = self.trainer.predict(
            self.model,
            dataloaders=dataloader
            if dataloader
            else self.datamodule.predict_dataloader(),
            ckpt_path=ckpt_path,
        )

        # Filter out None values from the predictions and true values
        predictions_batched, truevalue_batched = zip(
            *[(pred, true) for pred, true in out if pred is not None]
        )

        # Check if predictions_batched is a list and convert it to a tensor if so
        if isinstance(predictions_batched, list):
            predictions_batched = torch.vstack(predictions_batched)
        elif isinstance(predictions_batched, torch.Tensor):
            # If predictions_batched is already a tensor, no action needed
            pass
        elif isinstance(predictions_batched, tuple):
            print(
                "type of first element of predictions_batched: ",
                type(predictions_batched[0]),
            )
            # If predictions_batched is a tuple, convert it to a tensor
            predictions_batched = torch.vstack(predictions_batched)
        else:
            # Raise an error if predictions_batched is neither a tensor nor a list
            print("predictions_batched: ", predictions_batched)
            raise TypeError(
                f"Expected predictions_batched to be a list or a torch.Tensor, but got {type(predictions_batched)}"
            )

        if truevalue_batched is not None:
            if isinstance(truevalue_batched, list):
                truevalue_batched = torch.vstack(truevalue_batched)
            elif not isinstance(truevalue_batched, torch.Tensor):
                raise TypeError(
                    f"Expected truevalue_batched to be a list or a torch.Tensor, but got {type(truevalue_batched)}"
                )

        # Convert to numpy arrays
        stacked_predictions = predictions_batched.numpy()
        if truevalue_batched is not None:
            stacked_truevalues = truevalue_batched.numpy()
        else:
            stacked_truevalues = None

        print("Stacked predictions: ", stacked_predictions.shape)
        if stacked_truevalues is not None:
            print("Stacked true values: ", stacked_truevalues.shape)

        return stacked_predictions, stacked_truevalues

    def predict_embedding(self, dataloader=None, ckpt_path: str = None) -> np.ndarray:
        self._check_is_initialized()

        embeddings_list = []

        def capture_embedding(module, input, output):
            embeddings_list.append(output.detach())

        # Register hook at the layer just before the final layer
        # hook = self.model.classifier.layers[-2].register_forward_hook(capture_embedding)
        try:
            hook = self.model.base_model.classifier[14].register_forward_hook(
                capture_embedding
            )
        except AttributeError:
            try:
                hook = self.model.classifier[14].register_forward_hook(
                    capture_embedding
                )
            except AttributeError:
                print("Could not find classifier layer in model. Model: ", self.model)
                raise AttributeError

        # Make predictions (this will also trigger the hook)
        _ = self.trainer.predict(
            self.model,
            dataloaders=dataloader
            if dataloader
            else self.datamodule.predict_dataloader(),
            ckpt_path=ckpt_path,
        )

        # Remove hook
        hook.remove()

        return torch.vstack(embeddings_list).cpu().numpy()

    def predict_embedding_random_subset(
        self, dataloader=None, ckpt_path: str = None, subsample_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Like the predict_embedding method, but only uses a random subset of the data to compute the embedding.
        For this, re-initialize the dataloader with a random subset of the data. (i.e., sub_sample_frac < 1.0)
        Afterwards, the original dataloader is restored.
        """
        self._check_is_initialized()

        # Create new dataloader with random subset of the data
        self.init_datamodule(sub_sample_frac=subsample_ratio)
        new_dataloader = (
            self.datamodule.train_dataloader()
        )  # Only required for large train set

        # Get embedding
        embedding = self.predict_embedding(
            dataloader=new_dataloader, ckpt_path=ckpt_path
        )

        # Restore original dataloader
        self.datamodule.init_datamodule(sub_sample_frac=1.0)

        return embedding
