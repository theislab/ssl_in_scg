===================================
Bash Scripts for Masked Autoencoder
===================================

Overview
--------

This directory contains a collection of bash scripts designed to facilitate various data masking tasks within the context of computational biology, specifically focusing on masked autoencoder models for single-cell genomics data. The scripts are intended for use in high-performance computing environments and may require adaptation to meet the specific computational resources available to the user.

The primary purpose of these scripts is to enable the masking of individual genes, gene programs (GPs), and the conversion between gene programs and transcription factors (TFs) to enhance the analysis and interpretation of single-cell genomics data through the application of masked autoencoders.

Directory Structure
-------------------

The main subfolders within this directory are organized as follows:

- ``ind_gene_masking/``: Contains scripts for masking individual genes in the context of a masked autoencoder model.
- ``gp_to_tf/``: Houses the scripts required for masking in the process of converting gene programs to transcription factors (GP to TF).
- ``gp_to_gp/``: Includes scripts for masking in the context of gene program to gene program conversions.
- ``gene_program_masking/``: A collection of scripts dedicated to the masking of gene programs.

Each subfolder contains two types of bash scripts:

- ``MLPxyp.sh``: These scripts are tailored for the masked autoencoder on scTab with xy% of genes masked, facilitating the exploration of data under varying conditions of gene visibility.
- ``NegBin...``: Scripts for modeling the data as a negative binomial distribution using masked autoencoders, suitable for handling overdispersed count data typical in single-cell genomics.

Usage
-----

To utilize these scripts, navigate to the respective subfolder and select the script that corresponds to your specific data masking need. Prior to execution, it may be necessary to modify the script to align with the computational resources and environment specific to your setup. This may involve adjusting parameters related to memory usage, processing power, and the computational backend.

Please ensure that all dependencies and environmental variables are correctly configured before running any script. For detailed instructions on script modifications and execution, refer to the comments within each script file.
