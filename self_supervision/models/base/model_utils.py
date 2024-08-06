import torch.nn as nn


def build_encoder(
    n_gene: int,
    n_protein: int,
    n_batch: int,
    n_hidden: int,
    n_latent: int,
    n_layer: int,
    dropout: int,
    mode: str,
):
    if mode == "wbatch":
        modules = [
            nn.Linear(n_gene + n_protein + n_batch, n_hidden, bias=True),
            nn.BatchNorm1d(
                n_hidden,
                momentum=0.01,
                eps=0.001,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
        ]
        n_layer -= 1
        while n_layer > 0:
            modules += [
                nn.Linear(n_hidden + n_batch, n_hidden, bias=True),
                nn.BatchNorm1d(
                    n_hidden,
                    momentum=0.01,
                    eps=0.001,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout, inplace=False),
            ]
            n_layer -= 1
        modules.append(nn.Linear(n_hidden + n_batch, n_latent, bias=True))
        return nn.Sequential(*modules)
    elif mode == "wobatch":
        modules = [
            nn.Linear(n_gene + n_protein, n_hidden, bias=True),
            nn.BatchNorm1d(
                n_hidden,
                momentum=0.01,
                eps=0.001,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
        ]
        n_layer -= 1
        while n_layer > 0:
            modules += [
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.BatchNorm1d(
                    n_hidden,
                    momentum=0.01,
                    eps=0.001,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout, inplace=False),
            ]
            n_layer -= 1
        modules.append(nn.Linear(n_hidden, n_latent, bias=True))
        return nn.Sequential(*modules)
    else:
        print("wrong mode")


def build_decoder(
    n_gene: int,
    n_protein: int,
    n_batch: int,
    n_hidden: int,
    n_latent: int,
    n_layer: int,
    dropout: int,
    mode: str,
):
    if mode == "wbatch":
        modules = [
            nn.Linear(n_latent + n_batch, n_hidden, bias=True),
            nn.BatchNorm1d(
                n_hidden,
                momentum=0.01,
                eps=0.001,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
        ]
        n_layer -= 1
        while n_layer > 0:
            modules += [
                nn.Linear(int(n_hidden + n_batch), n_hidden, bias=True),
                nn.BatchNorm1d(
                    n_hidden,
                    momentum=0.01,
                    eps=0.001,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout, inplace=False),
            ]
            n_layer -= 1
        modules.append(
            nn.Linear(
                int(n_hidden + n_batch), int(n_gene + n_protein + n_batch), bias=True
            )
        )
        return nn.Sequential(*modules)
    elif mode == "wobatch":
        modules = [
            nn.Linear(n_latent, n_hidden, bias=True),
            nn.BatchNorm1d(
                n_hidden,
                momentum=0.01,
                eps=0.001,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
        ]
        n_layer -= 1
        while n_layer > 0:
            modules += [
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.BatchNorm1d(
                    n_hidden,
                    momentum=0.01,
                    eps=0.001,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout, inplace=False),
            ]
            n_layer -= 1
        modules.append(nn.Linear(n_hidden, n_gene + n_protein, bias=True))
        return nn.Sequential(*modules)
    else:
        print("wrong mode")
