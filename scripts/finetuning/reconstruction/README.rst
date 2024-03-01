===================================
Fine-Tuning: Reconstruction Scripts 
===================================

Overview
--------

This directory provides bash scripts designed for the fine-tuning phase of reconstructing single-cell transcriptomics data. 

Directory Structure
-------------------

### Principal Component Analysis

- ``pca.sh``: A script for computing the PCA of scTab data, serving as a foundational step for dimensionality reduction and data exploration.

### scTab Training Scripts

The following subfolders contain scripts for training the reconstruction models, with options for utilizing pre-trained models or starting without self-supervised learning (SSL) pretraining:

- ``cellnet_bt/``: Scripts for training with a pre-trained Barlow Twins model.
- ``cellnet_byol/``: Scripts for training with a pre-trained BYOL model.
- ``cellnet_gene_program_masking/``: Scripts for gene program masking-based training.
- ``cellnet_gp_to_gp/``: Scripts for gene program to gene program reconstruction.
- ``cellnet_gp_to_tf/``: Scripts for gene program to transcription factor reconstruction.
- ``cellnet_ind_gene_masking/``: Scripts for individual gene masking-based training.
- ``cellnet_no_ssl/``: Scripts for training without SSL pretraining.

Each of these folders contains:

- ``MLP.sh``: Bash scripts for training an autoencoder with a mean-squared-error loss function.
- ``NegBin.sh``: Bash scripts for training an autoencoder aimed at modeling the parameters of a negative binomial distribution.

### Fine-Tuning on Smaller Datasets / Cell Atlases

For fine-tuning models on specific datasets or cell atlases, the directory includes:

- ``hlca_no_ssl/``, ``hlca_gp_to_tf/``: Scripts for the Human Lung Cell Atlas.
- ``pbmc_no_ssl/``, ``pbmc_gp_to_tf/``: Scripts for PBMC datasets.
- ``tabula_sapiens_no_ssl/``, ``tabula_sapiens_gp_to_tf/``: Scripts for the Tabula Sapiens Atlas.

These folders also contain ``MLP.sh`` and ``NegBin.sh`` scripts for mean-squared-error and negative binomial autoencoders, respectively.

### Data Integration Tasks

For tasks related to data integration, potentially leveraging pre-trained models or starting anew:

- ``integration_ind_gene_masking_shallow/``: Scripts for shallow integration tasks with individual gene masking.
- ``integration_ind_gene_masking/``: Scripts for deep integration tasks with individual gene masking.
- ``integration_no_ssl/``: Scripts for integration tasks without SSL pretraining.

These folders exclusively contain ``MLP.sh`` scripts for mean-squared-error based autoencoder training.

Usage
-----

To utilize these scripts, navigate to the desired subfolder based on your specific requirements, whether it involves PCA computation, model training for reconstruction, fine-tuning on smaller datasets, or data integration tasks. Adjust the script parameters as necessary to accommodate your computational environment and dataset specifics.

Ensure that your computational environment is suitably configured with all necessary dependencies, and consult the comments within each script for guidance on adjustments and execution.
