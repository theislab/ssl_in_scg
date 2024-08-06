from typing import List, Dict, Tuple
import numpy as np
import torch
import time


def read_gmt(gp_file: str) -> List[List[str]]:
    """
    Read .gmt file
    :param gp_file: .gmt file
    :return: gene_programs: list of gene programs, each element is a list of genes
    """
    # If 'True', converts genes' names from files and adata to uppercase for comparison.
    genes_use_upper = True
    # If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
    # and cuts the name to the first thirty symbols.
    clean = True
    # read gene programs
    files = gp_file
    files = [files] if isinstance(files, str) else files
    for file in files:
        with open(file) as f:
            # remove header
            p_f = [line.upper() for line in f] if genes_use_upper else f
            terms = [line.strip("\n").split() for line in p_f]
    if clean:
        # remove terms with less than 2 genes
        terms = [[term[0].split("_", 1)[-1][:30]] + term[1:] for term in terms if term]
    # remove first two elements of each term
    gene_programs = [term[2:] for term in terms]
    return gene_programs


def read_gmt_to_dict(gp_file: str) -> Dict[str, List[str]]:
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


def encode_gene_programs(
    var_names: List[str],
    gene_program: List[List[str]],
) -> np.ndarray:
    """
    Encode gene programs into a multi-hot encoded tensor.
    The rows of the tensor are the gene programs and the columns are the genes in the same order as the gene list.
    The element of the tensor at row (gene program) i and column (gene) j is 1 if the gene program contains the gene and 0 otherwise.
    If a gene in gene_program is not in the gene list, ignore it.
    :param required_tolerance: percentage of genes at least present in the dataset for a gene program to be included
    :param var_names: list of genes in the dataset
    :param gene_program: gene programs to encode
    :param gene: list of genes
    :return: encoded gene programs
    """
    # get gene programs
    start_time = time.time()
    # create blank array
    encoded_gene_program = np.zeros((len(gene_program), len(var_names)))
    # iterate over gene programs

    for i, program in enumerate(gene_program):
        # get indices of genes in the program that are also in the dataset
        encoded_gene_program[i, np.in1d(var_names, program)] = 1
    print(f"Encoding gene programs took {time.time() - start_time:.2f} seconds")
    return encoded_gene_program


def encode_gene_program_to_transcription_factor(
    var_names: List[str], gene_program: Dict[str, List[str]], required_tolerance: int
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Encode gene programs into a multi-hot encoded tensor.
    Return a dictionary with the transcription factor as key and a tuple of the encoded gene program and the transcription factor index as value.
    The rows of the tensor are the gene programs, and the columns are the genes in the same order as the gene list.
    The element of the tensor at row (gene program) i and column (gene) j is 1 if the gene program contains the gene and 0 otherwise.
    If a gene in gene_program is not in the gene list, ignore it.
    Only include gene programs in the encoded output if they have at most 'required_tolerance' genes missing in 'var_names'.
    :param var_names: list of selected genes
    :param gene_program: gene programs to encode (dictionary of transcription factor - gene program pairs)
    :param required_tolerance: percentage of genes at least present in the dataset for a gene program to be included
    :return: encoded_gene_program: Dict[(Tensor, Tensor)] - dictionary with the transcription factor as key and a tuple of the encoded gene program and the transcription factor index as value
    """
    start_time = time.time()
    out = {}
    # Iterate over gene programs
    filtered_gene_programs = []
    for i, (tf, program) in enumerate(gene_program.items()):
        # Count the number of missing genes in var_names for the current gene program as
        # percentage of the total number of genes in the gene program
        missing_genes = sum(gene not in var_names for gene in program) / len(program)

        # Include the gene program in the encoded output only if it has at least 'required_tolerance' missing genes
        if (1 - missing_genes) <= (required_tolerance / 100):
            # check how many gene programs are left
            filtered_gene_programs.append(program)

            # encode the gene program as a multi-hot encoded tensor
            program_indices = np.where(np.isin(var_names, program))[0]
            encoded_gene_program = torch.zeros(len(var_names))
            encoded_gene_program[program_indices] = 1

            # continue if the gene program is empty
            if encoded_gene_program.sum() == 0:
                continue

            # encode the corresponding transcription factor as a one-hot encoded tensor
            tf_index_in_var_names = np.where(np.isin(var_names, tf))[0]
            tf_index_one_hot = torch.zeros(len(var_names))
            tf_index_one_hot[tf_index_in_var_names] = 1

            # store both as a tuple in the dictionary
            out[tf] = (encoded_gene_program, tf_index_one_hot)

    # Calculate the percentage and number of gene programs left after filtering
    original_count = len(gene_program.keys())
    filtered_count = len(filtered_gene_programs)
    percentage_remaining = (filtered_count / original_count) * 100

    print(
        f"Filtering gene programs based on missing genes took {time.time() - start_time:.2f} seconds"
    )
    print(
        f"{percentage_remaining:.2f}% of gene programs remaining after filtering ({filtered_count}/{original_count})"
    )

    return out
