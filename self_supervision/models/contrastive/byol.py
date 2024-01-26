"""
Bootstrap your own latent code
Paper: https://arxiv.org/abs/2006.07733
Code base: https://github.com/lucidrains/byol-pytorch
This code adapts the BYOL method to include custom augmentations
"""

import copy
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F

# helper functions
from typing import Optional, Callable, Union, Any


def default(val: Optional[Any], def_val: Optional[Any] = None) -> Optional[Any]:
    """Return def_val if val is None, otherwise val.
    Args:
        val: the value to be checked
        def_val: the default value to be returned if val is None
    Returns:
        val if val is not None, otherwise def_val
    """
    return def_val if val is None else val


# don't need flatten for tabular data
def flatten(t: torch.Tensor) -> torch.Tensor:
    """Flatten a given tensor.
    Args:
        t: the tensor to be flatten
    Returns:
        The flatten version of input tensor t
    """
    return t.reshape(t.shape[0], -1)


def singleton(cache_key: str) -> Callable:
    """Decorator to make a function return the same singleton instance.
    Args:
        cache_key: name of the attribute to use as a cache
    Returns:
        A wrapper function
    """

    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module: torch.nn.Module) -> torch.device:
    """
    This function takes in an instance of a Pytorch nn.Module,
    iterates over its parameters using module.parameters()
    and returns the device (CPU or GPU) of the first parameter using .device attribute.
    :param module: torch module
    :return:
    """
    return next(module.parameters()).device


def set_requires_grad(model: torch.nn.Module, val: bool):
    """
    Given a Pytorch model, this function sets the requires_grad attribute for all model parameters.
    :param model: The Pytorch model
    :param val: The value to set requires_grad to.
    """
    for param in model.parameters():
        param.requires_grad = val


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


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        """
        Apply a given function `fn` to the input data with probability `p`.
        The fn could be any transformation, such as random crop, flip, rotation, normalization etc.,
        which might be useful during data augmentation.

        Args:
            fn (Callable): A function to apply to the input data.
            p (float): The probability of applying the function.
        """
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average
class EMA:
    def __init__(self, beta):
        """
        Exponential moving average class.
        :param beta: The smoothing factor for the exponential moving average.
        """
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        """
        Updates the old value with the new value using the smoothing factor.
        :param old: The old value.
        :param new: The new value.
        :return: The updated value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    """
    Updates the moving average model with the current model's parameters using an EMA updater.
    :param ema_updater: An instance of the EMA class that holds the value of the smoothing factor 'beta'
    :param ma_model: The model holding the moving average parameters
    :param current_model: The current model whose parameters will be used to update the moving average
    :return: None
    """
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def MLP(dim, projection_size, hidden_size=4096):
    """
    MLP class for projector and predictor.
    :param dim: dimension of input
    :param projection_size: dimension of output
    :param hidden_size: hidden size
    :return: nn.Sequential
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


def SimSiamMLP(dim, projection_size, hidden_size=4096):
    """
    MLP class for projector and predictor for SimSiam.
    :param dim: dimension of input
    :param projection_size: dimension of output
    :param hidden_size: hidden size
    :return: nn.Sequential
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False),
    )


class NetWrapper(nn.Module):
    def __init__(
        self,
        net,
        projection_size: int,
        projection_hidden_size: int,
        layer: Union[int, str] = -2,
        use_simsiam_mlp: bool = False,
    ):
        """
        A wrapper class for the base neural network which manages the interception of the hidden layer output
        and pipes it into the projector and predictor nets.
        :param net: the base neural network which provides the hidden representation
        :param projection_size: the size of the final output tensor of the projector network
        :param projection_hidden_size: the size of the hidden layer of the projector network
        :param layer: the index or name of the layer of the base network whose output will be considered as the hidden representation
        :param use_simsiam_mlp: whether to use the SimSiamMLP or MLP class to create the projector network
        """
        super(NetWrapper, self).__init__()
        self.net = net
        self.layer = layer
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.use_simsiam_mlp = use_simsiam_mlp
        self.hidden = {}  # will store the output of the hidden layer
        self.hook_registered = False

    def _find_layer(self) -> nn.Module:
        """
        Finds the hidden layer of the base network using its index or name
        :return: the hidden layer module
        """
        if isinstance(self.layer, str):
            # if layer is specified by name, get it from the named_modules of the net
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif isinstance(self.layer, int):
            # if layer is specified by index, get it from the children of the net
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input_, output):
        """
        Store the forward hook by storing the flattened output in the hidden representation
        :param _:
        :param input_:
        :param output:
        :return:
        """
        device = input_[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        """
        Registers a forward hook on the hidden layer
        """
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(
            self._hook
        )  # we need this handle to remove the hook later  # noqa: F841
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        """
        returns the projector network to be used
        """
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(
            dim, self.projection_size, self.projection_hidden_size
        )
        return projector.to(hidden)

    def get_representation(self, x):
        """
        Get the hidden representation of x
        :param x: image
        :return: hidden representation
        """
        if self.layer == -1:
            out = self.net(x)
            if type(out) is tuple:  # tabnet
                output, loss = out
            else:
                output = out
            return output

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        out = self.net(x)
        if type(out) is tuple:  # tabnet
            _, _ = out
        else:
            _ = out
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x, return_projection=True):
        """
        forward function for BYOL
        :param x: input datapoint
        :param return_projection: If False, only representation is returned, if False, projection and representation are
        returned
        :return: representation / projection
        """
        representation = self.get_representation(x)
        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class GaussianBlur(object):
    def __init__(self, p: float):
        """
        Initialize the GaussianBlur class with standard deviation of the gaussian noise
        :param p: (float) the standard deviation of the gaussian noise
        """
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to the input image
        :param img: input image
        :return: image after applying gaussian blur
        """
        # create new tensor with same shape as img and random noise with mean 0 and std = self.p
        # this one results in the error 'TypeError: random_() received an invalid combination of arguments - got (std=float, mean=int, )'
        # new_img = torch.tensor(img) + torch.tensor(img).new(img.shape).random_(mean=0, std=self.p)
        # this one works
        new_img = img.clone().detach().requires_grad_(True) + img.clone().detach().new(
            img.shape
        ).normal_(mean=0, std=self.p)
        # Clamp the image pixel values between 0 and 1
        return torch.clamp(new_img, min=0)


class UniformBlur(object):
    def __init__(self, p: float):
        """
        Initialize the class with a float value p, which denotes the strength of the noise that is added.
        :param p: the strength of the noise that will be added.
        """
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        This function applies a uniform noise on the input image
        :param img: input image
        :return: returns the blurred image
        """
        # Create a new tensor of the same shape and dtype as the input image, and fill it with random noise
        # with uniform distribution between -p/2 and p/2
        new_img = img.clone().detach().requires_grad_(True) + img.clone().detach().new(
            img.shape
        ).uniform_(-self.p / 2, self.p / 2)
        # Clamp the values of the pixels of the image to be between 0 and 1
        return torch.clamp(new_img, min=0)


# main class
class BYOL(nn.Module):
    """
    BYOL implementation
    """

    def __init__(
        self,
        net,
        image_size,
        batch_size: int,
        augment_type: str,
        augment_intensity: float = 0.1,
        hidden_layer=-2,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        augment_fn2=None,
        moving_average_decay=0.99,
        use_momentum=True,
    ):
        super().__init__()
        self.net = net

        # augmentations for the two views
        if augment_type == "Gaussian":
            augment_f = GaussianBlur(augment_intensity)
        elif augment_type == "Uniform":
            augment_f = UniformBlur(augment_intensity)
        elif augment_type == "Meta_Cells":
            raise NotImplementedError

        # augmentations for the two views
        DEFAULT_AUG = augment_f

        print("Projection size: ", projection_size)
        print("Projection hidden size: ", projection_hidden_size)
        print("Hidden layer: ", hidden_layer)

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        # get device of network and make wrapper same device
        device = get_module_device(net)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(batch_size, image_size, device=device).to(device))

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, "you do not need to update the moving average, since you have turned off momentum for the target encoder"
        assert (
            self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(self, x, return_embedding=False, return_projection=True):
        assert not (
            self.training and x.shape[0] == 1
        ), "you must have greater than 1 sample when training, due to the batchnorm in the projection layer"

        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = (
                self._get_target_encoder() if self.use_momentum else self.online_encoder
            )
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
