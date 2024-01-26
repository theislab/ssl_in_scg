import os
import argparse
import pandas as pd
from os.path import join
import anndata
from typing import List, Tuple, Optional, Any
import dask.dataframe as dd
import numpy as np
import pickle
import logging
from contextlib import contextmanager
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)

DATA_DIR: str = (
    "/lustre/groups/ml01/workspace/till.richter/merlin_cxg_2023_05_15_sf-log1p"
)


@contextmanager
def managed_computation() -> Any:
    try:
        yield
    except MemoryError:
        logging.error("MemoryError: Releasing resources and continuing.")
        # Insert any code to free up resources, if needed
    except Exception as e:
        logging.error(str(e))
        return None


def get_count_matrix_and_obs(
    ddf: dd.DataFrame, perc: int = 100
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Less memory efficient but more reliable computation
    """
    if perc == 100:
        ddf_subsample = ddf
    else:
        fractions = [perc / 100, 1 - perc / 100]
        ddf_subsample, _ = ddf.random_split(fractions)

    x = ddf_subsample["X"].apply(sp.csr_matrix, meta=("X", "object")).compute()
    x = sp.vstack(x.values)
    obs = ddf_subsample[["cell_type", "tech_sample", "dataset_id"]].compute()

    return x, obs


def write_adata_with_scanpy_subsample(
    perc: int,
    hvg_indices: Optional[List[int]] = None,
    split: str = "train",
    adata_dir: str = "/lustre/groups/ml01/workspace/till.richter/",
):
    """
    Write AnnData object to disk with perc % of the data.
    :param perc: Percentage of data to use.
    :param hvg_indices: List of indices of highly variable genes.
    :param split: Data split, e.g. 'train', 'test'.
    :param adata_dir: Directory to save the AnnData object.
    :return: None
    """
    # Load the data
    ddf_split = dd.read_parquet(join(DATA_DIR, split))
    x_split, obs_split = get_count_matrix_and_obs(ddf_split, perc)

    gene_ens_ids = list(
        pd.read_parquet(os.path.join(DATA_DIR, "var.parquet"))["feature_id"]
    )

    # Compute and create AnnData object
    adata = anndata.AnnData(X=x_split, obs=obs_split, var={"ensembl_id": gene_ens_ids})
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1

    # Subset the data rows to hvg_indices
    if hvg_indices is not None:
        adata = adata[:, hvg_indices]

    # Write AnnData object to disk
    file_name = (
        f"cellxgene_{split}_adata.h5ad"
        if hvg_indices is None
        else f"cellxgene_hvg_{split}_adata.h5ad"
    )
    print(f"Writing AnnData object to {join(adata_dir, file_name)}")
    adata.write_h5ad(join(adata_dir, file_name))

    del adata, x_split, obs_split
    print("Done.")
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Write AnnData object to disk.")
    parser.add_argument(
        "--perc", type=int, default=100, help="Percentage of data to write"
    )
    parser.add_argument(
        "--hvg_indices", type=str, default=None, help="Path to HVG indices pickle file"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "test", "val"], help="Split name"
    )
    parser.add_argument("--adata_dir", type=str, default=".", help="Output directory")
    return parser.parse_args()


# run the script in bash with:
# python write_adata.py --perc 100 --split train --adata_dir /lustre/groups/ml01/workspace/mojtaba.bahrami/cellxgene/

if __name__ == "__main__":
    args = parse_arguments()
    if args.hvg_indices is not None:
        try:
            with open(args.hvg_indices, "rb") as f:
                hvg_indices = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"File {args.hvg_indices} does not exist.")
            exit(1)
    else:
        hvg_indices = None

    print(f"Writing adata for {args.perc}% of the data in {args.split}...")
    with managed_computation():
        write_adata_with_scanpy_subsample(
            perc=args.perc,
            hvg_indices=hvg_indices,
            split=args.split,
            adata_dir=args.adata_dir,
        )
    print("Done.")
