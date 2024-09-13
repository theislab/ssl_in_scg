import pandas as pd
from os.path import join
import anndata
import scanpy as sc
import dask.dataframe as dd
import numpy as np
import pickle


DATA_DIR = "/lustre/groups/ml01/workspace/till.richter/merlin_cxg_2023_05_15_sf-log1p"


def get_count_matrix_and_obs(ddf):
    x = (
        ddf["X"]
        .map_partitions(
            lambda xx: pd.DataFrame(np.vstack(xx.tolist())),
            meta={col: "f4" for col in range(19331)},
        )
        .to_dask_array(lengths=[1024] * ddf.npartitions)
    )
    obs = ddf[["cell_type", "tech_sample"]].compute()

    return x, obs


def get_hvg_list(perc: int, hvgs: int):
    ddf_train = dd.read_parquet(join(DATA_DIR, "train"), split_row_groups=True)
    x_train, obs_train = get_count_matrix_and_obs(ddf_train)
    print("Train data: ", x_train.shape)

    total_rows = x_train.shape[0]
    rows_to_select = int(total_rows * (perc / 100))
    random_indices = np.random.choice(total_rows, size=rows_to_select, replace=False)
    x_train_sub = x_train[random_indices,]

    print("Subsampled data: ", x_train_sub.shape)

    # Create a boolean mask to select the desired rows
    mask = np.zeros(len(obs_train), dtype=bool)
    mask[random_indices] = True
    obs_train_sub = obs_train.iloc[mask]

    adata = anndata.AnnData(X=x_train_sub, obs=obs_train_sub)

    out = sc.pp.highly_variable_genes(adata, n_top_genes=hvgs, inplace=False)

    hvg_indices = list(out.loc[out["highly_variable"]].index)

    with open("hvg_" + str(hvgs) + "_indices", "wb") as f:
        pickle.dump(list(hvg_indices), f)

    return hvg_indices


if __name__ == "__main__":
    # check with os if hvg_indices.pickle exists
    perc = 15
    for hvgs in [2000, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        try:
            hvg_indices = get_hvg_list(perc=perc, hvgs=hvgs)
        except Exception as e:
            print(e)
            continue
        print("Found HVGs using ", perc, "% of the data")

    print("Done")
