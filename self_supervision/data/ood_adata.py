###################
# For OOD testing #
###################

import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
import os
import h5py


def create_ood_dataloader(
    adata_path: str,
    data_path: str,
    batch_size: int = 32,
    write_to_dis: bool = True,
) -> DataLoader:
    """
    Create a DataLoader given an AnnData object.

    Parameters:
    - adata: AnnData object containing the dataset
    - data_path: General data path for other required files
    - batch_size: Batch size for DataLoader (default is 32)

    Returns:
    - DataLoader object
    """

    # Create CustomTensorDataset
    class CustomTensorDataset(Dataset):
        def __init__(self, tensor_x, tensor_y):
            self.tensor_x = tensor_x
            self.tensor_y = tensor_y

        def __getitem__(self, index):
            x_data = self.tensor_x[index]
            y_data = self.tensor_y[index]
            return {"X": x_data, "cell_type": y_data}

        def __len__(self):
            return len(self.tensor_x)

    # Create paths
    # directory name of adata_path
    tensor_x_path = os.path.join(os.path.dirname(adata_path), "processed_tensor_x.pt")
    tensor_y_path = os.path.join(os.path.dirname(adata_path), "preocessed_tensor_y.pt")

    # Check if tensor_x and tensor_y already exist
    # If they do, load them and return a DataLoader
    # If they don't, create them and return a DataLoader
    if os.path.exists(tensor_x_path) and os.path.exists(tensor_y_path):
        tensor_x = torch.load(tensor_x_path)
        tensor_y = torch.load(tensor_y_path)
        print("Loaded tensor_x and tensor_y from disk.")

        # Create DataLoader
        dataset = CustomTensorDataset(tensor_x, tensor_y)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print("Created DataLoader.")

        return test_loader

    # If they don't exist, create them
    # Load adata
    adata = sc.read_h5ad(adata_path)
    print("Loaded raw adata with shape: ", adata.X.shape)

    # Load CellNet genes
    cellnet_genes_path = os.path.join(data_path, "var.parquet")
    cellnet_genes = list(pd.read_parquet(cellnet_genes_path)["feature_id"])
    print("Loaded CellNet genes.")

    # Find common and missing genes
    if "Gene" in adata.var.columns:
        common_genes = list(set(adata.var["Gene"].index) & set(cellnet_genes))
        missing_genes = list(set(cellnet_genes) - set(adata.var["Gene"].index))
        # Create a dictionary to map 'Gene' to 'ensembl_ids'
        gene_to_ensembl = dict(zip(adata.var["Gene"].index, adata.var_names))
    elif "feature_name" in adata.var.columns:
        common_genes = list(set(adata.var["feature_name"].index) & set(cellnet_genes))
        missing_genes = list(set(cellnet_genes) - set(adata.var["feature_name"].index))
        # Create a dictionary to map 'feature_name' to 'ensembl_ids'
        gene_to_ensembl = dict(zip(adata.var["feature_name"].index, adata.var_names))
    else:
        raise ValueError("adata.var does not contain 'Gene' or 'feature_name' columns.")

    print(
        f"Found {len(common_genes)} common genes and {len(missing_genes)} missing genes."
    )

    # Convert common genes to their corresponding ensembl IDs
    common_ensembl_ids = [gene_to_ensembl[gene] for gene in common_genes]

    # Filter and reorder genes
    adata = adata[:, common_ensembl_ids]
    print("Filtered and reordered genes. New adata shape: ", adata.X.shape)

    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Normalized and log-transformed data.")

    # Load cell_type_mapping
    cell_type_mapping = pd.read_parquet(
        os.path.join(data_path, "categorical_lookup/cell_type.parquet")
    )

    # Create mapping dictionary for cell_type to int64 encoding
    cell_type_to_encoding = {
        cell_type: idx for idx, cell_type in cell_type_mapping["label"].items()
    }

    # Filter cells by valid cell types
    valid_cell_types = set(cell_type_to_encoding.keys())
    adata = adata[adata.obs["cell_type"].isin(valid_cell_types)]
    print("Filtered cells by valid cell types. New adata shape: ", adata.X.shape)

    # Encode cell types
    # y_adata = np.array([cell_type_to_encoding[cell_type] for cell_type in adata.obs['cell_type'].values])

    # Zero-padding
    if missing_genes:
        zero_padding_df = pd.DataFrame(
            data=0, index=adata.obs.index, columns=missing_genes
        )

        concatenated_df = pd.concat([adata.to_df(), zero_padding_df], axis=1)
        concatenated_df = concatenated_df[cellnet_genes]  # Ensure ordering of genes

        # Create new AnnData object to ensure consistency
        adata = sc.AnnData(
            X=concatenated_df.values,
            obs=adata.obs,
            var=pd.DataFrame(index=cellnet_genes),
        )

    # Double-check that the genes are in the correct order
    assert all(adata.var_names == cellnet_genes), "Genes are not in the correct order."

    print("Final shape of adata: ", adata.X.shape)

    # PyTorch DataLoader
    # Assuming you have a function called `cell_type_to_encoding` to convert cell_type to int64
    tensor_x = torch.Tensor(adata.X)
    tensor_y = torch.Tensor(
        adata.obs["cell_type"].map(cell_type_to_encoding).values
    ).type(torch.int64)

    # Write to disk
    if write_to_dis:
        torch.save(tensor_x, tensor_x_path)
        torch.save(tensor_y, tensor_y_path)

    # Assertions to check functionality
    assert (
        tensor_x.shape[0] == tensor_y.shape[0]
    ), "Mismatch between number of samples and labels"

    # Create DataLoader
    dataset = CustomTensorDataset(tensor_x, tensor_y)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Created DataLoader.")

    return test_loader


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, "r") as hdf5_file:
            self.length = len(hdf5_file["cell_type"])

    def __getitem__(self, index):
        with h5py.File(self.file_path, "r") as hdf5_file:
            x_data = hdf5_file["X"][index]
            y_data = hdf5_file["cell_type"][index]
        # Convert to PyTorch tensors, if necessary
        x_data = torch.from_numpy(x_data).float()
        y_data = torch.from_numpy(y_data).long()
        return {"X": x_data, "cell_type": y_data}

    def __len__(self):
        return self.length
