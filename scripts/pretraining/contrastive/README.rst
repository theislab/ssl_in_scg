=====================================
Contrastive Pretraining Bash Scripts
=====================================

Overview
--------

This directory is dedicated to bash scripts for contrastive pretraining. The scripts employ two distinct contrastive learning frameworks: Bootstrap Your Own Latent (BYOL) and Barlow Twins (BT), each designed to enhance data representation and feature extraction in an unsupervised manner.

Directory Structure
-------------------

The directory is organized into two subfolders, each corresponding to a specific contrastive pretraining method:

- ``BYOL/``: Contains a single bash script for executing the BYOL pretraining process.
- ``BarlowTwins/``: Houses a bash script dedicated to running the Barlow Twins pretraining methodology.

Both scripts are crafted to facilitate the application of their respective pretraining frameworks, enabling users to improve the quality of data representations extracted from single-cell genomics datasets.

Usage
-----

To use these scripts, navigate to the corresponding subfolder for the desired pretraining framework. Each subfolder contains a single bash script (`BYOL.sh` for BYOL and `BT.sh` for Barlow Twins) that can be executed to start the pretraining process.

Before running the script, it may be necessary to adjust certain parameters or configurations to suit your specific computational environment and resources. This could include settings related to GPU utilization, batch sizes, or learning rates. Detailed comments within each script provide guidance on how to make these adjustments.

Please ensure your environment is properly set up with all necessary dependencies and configurations before attempting to run the script.
