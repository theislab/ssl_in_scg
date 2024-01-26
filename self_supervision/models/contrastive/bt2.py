# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adaptions from original implementation:
- Uses MLP instead of ResNet50 as Backbone
- Different transformations
"""
import math
import os
from torch import nn, optim
import torch


def adjust_learning_rate(args, optimizer, loader_length, step):
    """
    Adjusts the learning rate for the current step.
    Changed from loader to directly pass in loader_length
    :param args:
    :param optimizer:
    :param loader_length:
    :param step:
    :return:
    """
    max_steps = args.epochs * loader_length
    warmup_steps = 10 * loader_length
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
    optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(
        self,
        backbone,
        args,
        dropout=0.0,
        mode="cellnet",  # cellnet or multiomics
    ):
        super().__init__()
        self.args = args

        # for multiomics
        self.mode = mode
        self.n_genes = 2000
        self.n_proteins = 134
        self.n_batches = 12
        self.n_latent = 40
        self.n_projector_dim = 512
        self.n_hidden = 256
        self.dropout = dropout

        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        if self.mode == "multiomics":
            self.backbone = nn.Sequential(
                nn.Linear(
                    in_features=self.n_genes + self.n_proteins + self.n_batches,
                    out_features=self.n_hidden,
                ),
                nn.BatchNorm1d(self.n_hidden),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(
                    in_features=self.n_hidden + self.n_batches,
                    out_features=self.n_hidden,
                ),
                nn.BatchNorm1d(self.n_hidden),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(
                    in_features=self.n_hidden + self.n_batches,
                    out_features=self.n_latent,
                ),
            )
        else:
            self.backbone = backbone
        self.backbone.fc = nn.Identity
        self.distributed = False  # hardcoded for now

        # projector
        sizes = [64] + list(map(int, args.projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        if self.mode == "multiomics":
            self.projector = nn.Sequential(
                nn.Linear(
                    in_features=self.n_latent + self.n_batches,
                    out_features=self.n_latent,
                ),
                nn.BatchNorm1d(self.n_latent),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.n_latent + self.n_batches,
                    out_features=self.n_latent,
                ),
                nn.BatchNorm1d(self.n_latent),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.n_latent + self.n_batches,
                    out_features=self.n_projector_dim,
                    bias=False,
                ),
                nn.BatchNorm1d(self.n_projector_dim),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.n_projector_dim + self.n_batches,
                    out_features=self.n_projector_dim,
                    bias=False,
                ),
            )
        else:
            self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        if self.mode == "multiomics":
            covariate1 = y1[:, 2134:]
            covariate2 = y2[:, 2134:]
            for i in range(0, len(self.backbone)):
                if (i % 4 == 0) & (i != 0):
                    z1 = torch.cat((y1, covariate1), dim=1)
                    z1 = self.backbone[i](z1)
                    z2 = torch.cat((y2, covariate2), dim=1)
                    z2 = self.backbone[i](z2)
                else:
                    z1 = self.backbone[i](z1)
                    z2 = self.backbone[i](z2)
            for i in range(0, len(self.projector)):
                if i % 3 == 0:
                    z1 = torch.cat((z1, covariate1), dim=1)
                    z1 = self.projector[i](z1)
                    z2 = torch.cat((z2, covariate2), dim=1)
                    z2 = self.projector[i](z2)
                else:
                    z1 = self.projector[i](z1)
                    z2 = self.projector[i](z2)
        else:
            z1 = self.projector(self.backbone(y1))
            z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus if multiple gpus are used
        if self.distributed:
            c.div_(self.args.batch_size)
            torch.distributed.all_reduce(c)
        else:
            c = c / self.args.batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=False,
        lars_adaptation_filter=False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


"""
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
"""


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
        # new_img = torch.tensor(img) + torch.tensor(img).new(img.shape).random_(mean=0, std=self.p)
        # this one works
        new_img = img.clone().detach().requires_grad_(True) + img.clone().detach().new(
            img.shape
        ).normal_(mean=0, std=self.p)
        # Clamp the image pixel values between 0 and 1
        return torch.clamp(new_img, min=0)


class Transform:
    def __init__(self, p: float):
        """
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        """
        # Instead of images we have a vector and hence we only apply the GaussianBlur
        self.transform = GaussianBlur(p=p)
        self.transform_prime = GaussianBlur(p=0.1 * p)

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
