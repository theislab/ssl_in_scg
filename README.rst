Delineating the Effective Use of Self-Supervised Learning in Single-Cell Genomics
=================================================================================

Repository for the `paper <https://doi.org/10.1101/2024.02.16.580624>`_.

System Requirements
-------------------
- Python 3.10
- Dependencies listed in `requirements.txt`

Installation Guide
-------------------
1. Create a conda environment:

   .. code-block:: bash

       conda env create -f environment.yml

2. Activate the environment:

   .. code-block:: bash

       conda activate ssl

3. Install the package in development mode:

   .. code-block:: bash

       cd directory_where_you_have_your_git_repos/ssl_in_scg
       pip install -e .

4. Create symlink to the storage folder for experiments:

   .. code-block:: bash

       cd directory_where_you_have_your_git_repos/ssl_in_scg
       ln -s folder_for_experiment_storage project_folder

Demo
----

**Large Dataset:**

For large datasets, use the store-creation notebooks in the `scTab repository <https://github.com/theislab/scTab/tree/main/notebooks/store_creation>`_ to create a Merlin datamodule for efficient data loading.

**Small Dataset or Single Adata Object:**

For small datasets or a single Adata object, a simple PyTorch dataloader suffices. Refer to our `multiomics application <https://github.com/theislab/ssl_in_scg/blob/master/self_supervision/data/datamodules.py#L173>`_. A minimal example for masked pre-training of a smaller adata object is available in `sc_mae <https://github.com/theislab/sc_mae>`_.

**Expected output:**

Running the models will generate a checkpoint file with trained model parameters, saved using PyTorch Lightning's checkpointing functionality. This file can be used for inference, further training, or reproducibility.

**Expected run time:**

We pre-trained on a single GPU for approximately 1-2 days and fine-tuned on a single GPU about 12-24 hours. This depends, among others, on the underlying architecture, dataset, and hyperparameters. So, convergence should be watched.

Model checkpoints
-----------------
Pre-trained model checkpoints are available on `Hugging Face <https://huggingface.co/TillR/sc_pretrained>`_.

Retraining
----------

Obtain the dataset from the `scTab repository <github.com/theislab/scTab>`_ or write a Merlin store on your custom data. Then change `DATA_DIR` in `paths.py` to your custom dataset or keep it with the scTab dataset. After that, follow the scripts for pre-training and fine-tuning.

Citation
--------

If you find our work useful, please cite the following paper:

**Delineating the Effective Use of Self-Supervised Learning in Single-Cell Genomics**

`Link to the paper <https://doi.org/10.1101/2024.02.16.580624>`_

If you use the scTab data in your research, please cite the following paper:

**Scaling cross-tissue single-cell annotation models**

`Link to the paper <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10592700/>`_

Licence
-------
`self_supervision` is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

Authors
-------

`ssl_in_scg` was written by `Till Richter <till.richter@helmholtz-muenchen.de>`_, `Mojtaba Bahrami <mojtaba.bahrami@helmholtz-muenchen.de>`_, `Yufan Xia <yufan.xia@helmholtz-muenchen.de>`_ and `Felix Fischer  <felix.fischer@helmholtz-muenchen.de>`_ .
