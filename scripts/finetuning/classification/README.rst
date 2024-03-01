================================
Cell Type Classification Scripts
================================

Overview
--------

This directory houses a collection of bash scripts designed for the task of cell type classification within single-cell transcriptomics data. These scripts facilitate the classification of cell types across various datasets, including scTab, the Human Lung Cell Atlas (HLCA), PBMC datasets, and the Tabula Sapiens dataset. The scripts enable classification using a variety of methodologies, including training with and without pre-trained models and focusing on highly-variable gene (HVG) analysis.

Directory Structure
-------------------

### scTab Classification Scripts

For classification on the scTab dataset, with options for using pre-trained models or starting from scratch:

- ``cellnet_bt/``: Scripts for classification using a pre-trained Barlow Twins model.
- ``cellnet_byol/``: Scripts for classification using a pre-trained BYOL model.
- ``cellnet_gene_program_masking/``: Scripts focusing on gene program masking.
- ``cellnet_gp_to_gp/``: Scripts for gene program to gene program classification.
- ``cellnet_gp_to_tf/``: Scripts for gene program to transcription factor classification.
- ``cellnet_ind_gene_masking/``: Scripts for individual gene masking-based classification.
- ``cellnet_no_ssl/``: Scripts for classification without self-supervised learning pretraining.

### HLCA Cell Type Annotation

For cell type annotation on the Human Lung Cell Atlas:

- ``hlca_bt/``, ``hlca_byol/``, ``hlca_gene_program_masking/``, ``hlca_gp_to_gp/``, ``hlca_gp_to_tf/``, ``hlca_ind_gene_masking/``, ``hlca_no_ssl/``: Each folder contains scripts tailored to the HLCA dataset, with methodologies paralleling those used for the scTab dataset.

### PBMC Dataset Annotation

For cell type annotation on the PBMC dataset:

- ``pbmc_gene_program_masking/``, ``pbmc_ind_gene_masking/``, ``pbmc_no_ssl/``: Scripts specifically prepared for the classification and annotation tasks within the PBMC dataset.

### Tabula Sapiens Dataset Annotation

For cell type annotation on the Tabula Sapiens dataset:

- ``tabula_sapiens_gene_program_masking/``, ``tabula_sapiens_ind_gene_masking/``, ``tabula_sapiens_no_ssl/``: Contains scripts for classification tasks tailored to the Tabula Sapiens dataset, with a focus on gene program masking and individual gene masking.

### Highly-Variable Gene Analysis

For HVG analysis, focusing on classification across subselected genes:

- ``hvg_analysis_cellnet_ind_gene_masking/``, ``hvg_analysis_cellnet_no_ssl/``: Scripts designed for classification tasks centered around the analysis of highly-variable genes.

Usage
-----

To use these scripts, navigate to the corresponding subfolder based on your dataset and methodological preference. It may be necessary to adjust script parameters to align with your computational resources and the specific characteristics of your dataset.

Ensure that your computational environment is adequately prepared with all necessary dependencies and configurations before executing any script. The comments within each script offer detailed guidance on parameter adjustments and execution procedures.
