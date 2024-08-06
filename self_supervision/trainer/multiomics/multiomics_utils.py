import numpy as np
import torch
import pandas as pd


# convert batch (categorical) into numerical
def one_hot_encode(labels, num_labels=12):
    unique_labels = np.unique(labels)  # Get unique labels

    num_samples = len(labels)

    one_hot_encoded = np.zeros((num_samples, num_labels), dtype=int)

    for i, label in enumerate(labels):
        label_index = np.where(unique_labels == label)[0][0]  # Find index of label
        one_hot_encoded[i, label_index] = 1  # Set the corresponding index to 1

    return torch.tensor(one_hot_encoded)


# filter the gene sets,
# default: at least the half of the genes in the gene set should be overlapped with the given dataset
# and the gene set should have at least 100 genes
def filter_gene_set(
    gene_set_all, min_gene_set_size: int = 100, min_overlap_pct: float = 0.5
) -> pd.DataFrame:
    gene_set = gene_set_all[gene_set_all["total_num"] >= min_gene_set_size]
    gene_set = gene_set[gene_set["overlap_pct"] >= min_overlap_pct]
    gene_set = gene_set.sort_values(by=["overlap_pct", "total_num"], ascending=False)
    gene_set.reset_index(inplace=True, drop=True)
    return gene_set


def encode_tf(var_name, tf_dict):
    overlap_tf = {}
    i = 0
    for key, value in tf_dict.items():
        genes = var_name
        set1 = set(genes)
        set2 = set(value)

        overlap = list(set1.intersection(set2))

        if (key in genes) & (len(overlap) >= 10):
            idx_tf = genes.index(key)
            overlap_tf[i] = (idx_tf, [])
            for gene in value:
                if gene in genes:
                    overlap_tf[i][1].append(genes.index(gene))
            i += 1
    return overlap_tf


def read_gmt_to_dict(gp_file: str):
    """
    Read .gmt file and output a dictionary of gene programs
    The additional dict key is required for the gene program to transcription factor prediction
    :param gp_file: .gmt file
    :return: gene_programs_dict: dictionary of gene programs, where the key is the name of the gene program
    """
    # If 'True', converts genes' names from files and adata to uppercase for comparison.
    genes_use_upper = True
    # If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
    # and cuts the name to the first thirty symbols.
    clean = False
    # read gene programs
    files = gp_file
    files = [files] if isinstance(files, str) else files
    gene_programs_dict = {}
    for file in files:
        with open(file) as f:
            # remove header
            p_f = [line.upper() for line in f] if genes_use_upper else f
            terms = [line.strip("\n").split() for line in p_f]
        if clean:
            # remove terms with less than 2 genes
            terms = [
                [term[0].split("_", 1)[-1][:30]] + term[1:] for term in terms if term
            ]
        # remove first two elements of each term
        # gene_programs = [term[2:] for term in terms]
        for term in terms:
            tf_name = term[0].split("_")[0]
            gene_programs_dict[tf_name] = term[2:]
    return gene_programs_dict


def encode_gene_program(overlapped_genes: np.ndarray, query_adata) -> np.ndarray:
    query_genes = query_adata.var.index.to_numpy()
    encoded_genes = np.zeros((len(overlapped_genes), len(query_genes)))
    for i, gene_set in enumerate(overlapped_genes):
        encoded_genes[i] = np.isin(query_genes, gene_set).astype(int)
    return encoded_genes


def load_enc_weights(ae_model, pretrained_dict):
    """
    Load encoder weights from a pretrained autoencoder model
    Disregard decoder weights
    """
    model_dict = ae_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict and "decoder" not in k
    }
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    ae_model.load_state_dict(model_dict, strict=False)
    return ae_model
