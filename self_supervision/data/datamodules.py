import os
from math import ceil
from os.path import join
from typing import Dict, List

import lightning.pytorch as pl
import merlin.io
from merlin.dataloader.torch import Loader
from merlin.dtypes import float32, int64, boolean
from merlin.schema import ColumnSchema, Schema
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import numpy as np

PARQUET_SCHEMA = {
    "X": float32,
    "soma_joinid": int64,
    "is_primary_data": boolean,
    "dataset_id": int64,
    "donor_id": int64,
    "assay": int64,
    "cell_type": int64,
    "development_stage": int64,
    "disease": int64,
    "tissue": int64,
    "tissue_general": int64,
    "tech_sample": int64,
    "idx": int64,
}


def _merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict):
    return merlin.io.Dataset(
        path,
        engine="parquet",
        schema=Schema(
            [
                ColumnSchema(
                    "X",
                    dtype=float32,
                    is_list=True,
                    is_ragged=False,
                    properties={"value_count": {"max": 19357}},
                )
            ]
            + [ColumnSchema(col, dtype=int64) for col in columns]
        ),
        **dataset_kwargs,
    )


def _set_default_kwargs_dataloader(kwargs: Dict[str, any], train: bool = True):
    if kwargs is None:
        kwargs = {}

    parts_per_chunk = 8 if train else 1
    drop_last = True if train else False
    shuffle = True if train else False

    if "parts_per_chunk" not in kwargs:
        kwargs["parts_per_chunk"] = parts_per_chunk
    if "drop_last" not in kwargs:
        kwargs["drop_last"] = drop_last
    if "shuffle" not in kwargs:
        kwargs["shuffle"] = shuffle

    return kwargs


def _set_default_kwargs_dataset(kwargs: Dict[str, any], train: bool = True):
    if kwargs is None:
        kwargs = {}

    part_size = "100MB" if train else "325MB"

    if all(["part_size" not in kwargs, "part_mem_fraction" not in kwargs]):
        kwargs["part_size"] = part_size

    return kwargs


def _get_data_files(base_path: str, split: str, sub_sample_frac: float):
    if sub_sample_frac == 1.0:
        # if no subsampling -> just return base path and merlin takes care of the rest
        return join(base_path, split)
    else:
        files = [
            file
            for file in os.listdir(join(base_path, split))
            if file.endswith(".parquet")
        ]
        files = [
            join(base_path, split, file)
            for file in sorted(files, key=lambda x: int(x.split(".")[1]))
        ]
        return files[: ceil(sub_sample_frac * len(files))]


class MerlinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        columns: List[str],
        batch_size: int,
        sub_sample_frac: float = 1.0,
        dataloader_kwargs_train: Dict = None,
        dataloader_kwargs_inference: Dict = None,
        dataset_kwargs_train: Dict = None,
        dataset_kwargs_inference: Dict = None,
        dataset_id_filter=None,
    ):
        super(MerlinDataModule).__init__()

        for col in columns:
            assert col in PARQUET_SCHEMA

        self.dataloader_kwargs_train = _set_default_kwargs_dataloader(
            dataloader_kwargs_train, train=True
        )
        self.dataloader_kwargs_inference = _set_default_kwargs_dataloader(
            dataloader_kwargs_inference, train=False
        )
        self.dataset_id_filter = dataset_id_filter

        self.train_dataset = _merlin_dataset_factory(
            _get_data_files(path, "train", sub_sample_frac),
            columns,
            _set_default_kwargs_dataset(dataset_kwargs_train, train=True),
        )
        self.val_dataset = _merlin_dataset_factory(
            _get_data_files(path, "val", sub_sample_frac),
            columns,
            _set_default_kwargs_dataset(dataset_kwargs_inference, train=False),
        )
        self.test_dataset = _merlin_dataset_factory(
            join(path, "test"),
            columns,
            _set_default_kwargs_dataset(dataset_kwargs_inference, train=False),
        )

        self.batch_size = batch_size

    def train_dataloader(self):
        return Loader(
            self.train_dataset,
            batch_size=self.batch_size,
            **self.dataloader_kwargs_train,
        )

    def val_dataloader(self):
        return Loader(
            self.val_dataset,
            batch_size=self.batch_size,
            **self.dataloader_kwargs_inference,
        )

    def test_dataloader(self):
        return Loader(
            self.test_dataset,
            batch_size=self.batch_size,
            **self.dataloader_kwargs_inference,
        )

    def predict_dataloader(self):
        return Loader(
            self.test_dataset,
            batch_size=self.batch_size,
            **self.dataloader_kwargs_inference,
        )


class MultiomicsDataloader(Dataset):
    def __init__(self, proteins, genes, batches):
        self.proteins = proteins
        self.genes = genes
        self.batches = batches

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        batch = {
            "X": self.genes[idx],
            "protein": self.proteins.iloc[idx].values,
            "batch": self.batches[idx],
        }
        return batch


class AdataDataset(Dataset):
    def __init__(self, genes, perturbations):
        self.genes = genes
        self.perturbations = perturbations

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        batch = {"X": self.genes[idx], "perturbations": self.perturbations[idx]}
        return batch


class HDF5Dataset(Dataset):
    """Custom Dataset for loading data from HDF5 files separately."""

    def __init__(self, x_file_path, y_file_path):
        self.x_file_path = x_file_path
        self.y_file_path = y_file_path

        # Save all labels into tensor_y
        self.tensor_y = []
        with h5py.File(y_file_path, "r") as hdf5_file:
            for i in range(len(hdf5_file["processed_tensor_y"])):
                self.tensor_y.append(hdf5_file["processed_tensor_y"][i])

        # Assuming that the length of processed_tensor_x and processed_tensor_y datasets are the same
        with h5py.File(x_file_path, "r") as hdf5_file:
            self.length = len(hdf5_file["processed_tensor_x"])

    def __getitem__(self, index):
        # Open the hdf5 files
        with h5py.File(self.x_file_path, "r") as hdf5_x_file, h5py.File(
            self.y_file_path, "r"
        ) as hdf5_y_file:
            # Extract data using the index
            x_data = hdf5_x_file["processed_tensor_x"][index]

            y_data = hdf5_y_file["processed_tensor_y"][index]

            # Make sure x_data is a float tensor and y_data is a long tensor
            # If y_data is just a scalar, wrap it in an array
            if isinstance(y_data, np.integer):
                y_data = np.array([y_data])

            x_data = torch.from_numpy(x_data).float()
            y_data = torch.from_numpy(y_data).long()

        return {"X": x_data, "cell_type": y_data}

    def __len__(self):
        return self.length


def get_large_ood_dataloader(
    x_file_path, y_file_path, batch_size, shuffle=False, num_workers=4
):
    """Function to get a DataLoader for large out-of-core datasets."""
    dataset = HDF5Dataset(x_file_path, y_file_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
