# from sfairazero

import collections
from math import sqrt
from typing import Callable, List, Optional, Type, Iterable
import lightning.pytorch as pl
import abc
import torch
import torch.nn as nn
import numpy as np
from typing import Dict


def adamw_optimizer(model_params: Iterable, **kwargs: Dict) -> torch.optim.AdamW:
    """
    Create an optimizer with AdamW algorithm, including amsgrad.
    :param model_params: the parameters of the model to optimize
    :param kwargs: the additional keyword arguments to pass to the optimizer
    :return: AdamW optimizer
    """
    return torch.optim.AdamW(model_params, **kwargs, amsgrad=True)


def is_last_layer(i: int, units: List[int]) -> bool:
    """
    Returns a boolean indicating whether the current layer is the last layer in the units list.

    Parameters:
        i (int): current index of the layer
        units (List[int]): list of units for each layer

    Returns:
        bool: True if the current layer is the last one, False otherwise
    """
    return i == (len(units) - 2)


def _get_norm_layer(batch_norm: bool, layer_norm: bool) -> Optional[Type[nn.Module]]:
    """
    Returns the normalization layer to use.

    Parameters:
        batch_norm (bool): whether to use BatchNorm
        layer_norm (bool): whether to use LayerNorm

    Returns:
        Optional[Type[nn.Module]]: the normalization layer, or None if no normalization is to be used
    """
    if batch_norm:
        norm_layer = nn.BatchNorm1d
    elif layer_norm:
        norm_layer = nn.LayerNorm
    else:
        norm_layer = None

    return norm_layer


class DenseResidualBlock(nn.Module):
    def __init__(
        self,
        n_features: int,
        activation: Callable[[], torch.nn.Module],
        gain_weight_init: float = sqrt(2.0),
    ):
        """
        A dense residual block as described in the paper "Densely Connected Convolutional Networks"

        Parameters:
            n_features (int): number of input and output features for the linear layer
            activation (Callable[[], torch.nn.Module]): callable function returning the activation function to use
            gain_weight_init (float): gain value to use for Xavier weight initialization
        """
        if n_features is None:
            raise ValueError("n_features must not be None")
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(n_features, n_features)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain_weight_init)
        self.activation = activation()

    def forward(self, x):
        """
        Forward pass of the dense residual block.

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.activation(self.linear1(x)) + x


class DenseLayerStack(nn.Module):
    def __init__(
        self,
        in_features: int,
        units: List[int],
        activation: Callable[[], torch.nn.Module],
        batch_norm: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0,
        gain_weight_init: float = sqrt(2.0),
        add_residual_blocks: bool = False,
    ):
        """
        A dense stack of layers, with option for batchnorm, layernorm, dropout, residual connections and weight initialization.

        Parameters:
            in_features (int): number of input features
            units (List[int]): list of units for each dense layer
            activation (Callable[[], torch.nn.Module]): callable function returning the activation function to use
            batch_norm (bool): whether to use batch normalization
            layer_norm (bool): whether to use layer normalization
            dropout (float): dropout rate, between 0 and 1
            gain_weight_init (float): gain value for Xavier weight initialization
            add_residual_blocks (bool): whether to add residual connections
        """
        super(DenseLayerStack, self).__init__()

        layers_dim = [in_features] + units
        layers = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            cell = [nn.Linear(n_in, n_out)]
            if gain_weight_init is not None:
                nn.init.xavier_uniform_(cell[0].weight, gain=gain_weight_init)
            if batch_norm and not is_last_layer(i, layers_dim):
                cell.append(nn.BatchNorm1d(n_out, eps=1e-2))
            if layer_norm and not is_last_layer(i, layers_dim):
                cell.append(nn.LayerNorm(n_out))
            if not is_last_layer(i, layers_dim):
                cell.append(activation())
                if add_residual_blocks:
                    cell.append(DenseResidualBlock(n_out, activation, gain_weight_init))
            if (dropout > 0.0) and not is_last_layer(i, layers_dim):
                cell.append(nn.Dropout(dropout))

            layers.append((f"Layer {i}", nn.Sequential(*cell)))

        self.layers = nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        """
        Forward pass of the dense stack of layers

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.layers(x)


class MLP(torch.nn.Sequential):
    """
    This class implements the multi-layer perceptron (MLP) module.
    It uses torch.nn.Sequential to make the forward call sequentially.
    Implementation slightly adapted from https://pytorch.org/vision/main/generated/torchvision.ops.MLP.html
    (removed Dropout from last layer + log_api_usage call)

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear
        layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of
         the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer wont be
         used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place.
        Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        dropout: float = 0.0,
        final_activation: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim, eps=0.001))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        # the last layer should not have dropout
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        if final_activation is not None:
            layers.append(final_activation())

        super().__init__(*layers)


class ContrastiveBaseModel(pl.LightningModule, abc.ABC):
    """
    The same as BaseModel just without any optimization, since that's done through contrastive training
    """

    def __init__(
        self,
        gene_dim: int,
        feature_normalizations: List[str],
        feature_means: np.ndarray = None,
    ):
        super(ContrastiveBaseModel, self).__init__()
        for norm in feature_normalizations:
            if norm not in ["log1p", "zero_center", "none"]:
                raise ValueError(
                    f"Feature normalizations have to be in ['log1p', 'zero_center', 'none']. "
                    f"You supplied: {norm}"
                )
        if "zero_center" in feature_normalizations:
            if feature_means is None:
                raise ValueError(
                    'You need to supply feature_means to use "zero_center" normalization'
                )
            if not feature_means.shape == (1, gene_dim):
                raise ValueError("Shape of feature_means has to be (1, gene_dim)")
            self.register_buffer("feature_means", torch.tensor(feature_means))

    def _get_normalized_counts(self, x, normalization: str) -> torch.Tensor:
        if normalization == "log1p":
            x_normed = torch.log1p(x["X"])
        elif normalization == "zero_center":
            x_normed = x["X"] - self.feature_means
        else:
            x_normed = x["X"]

        return x_normed

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if ("labels" in batch) and (batch["labels"].shape[1] == 1):
            batch["labels"] = torch.squeeze(batch["labels"])
        if "batch" in batch:
            batch["batch"] = torch.squeeze(batch["batch"])
        if "assay_sc" in batch:
            batch["assay_sc"] = torch.squeeze(batch["assay_sc"])
        if "organ" in batch:
            batch["organ"] = torch.squeeze(batch["organ"])

        return batch
