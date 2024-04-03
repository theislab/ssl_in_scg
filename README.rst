Delineating the Effective Use of Self-Supervised Learning in Single-Cell Genomics
=================================================================================

Repository for the `paper <https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1.abstract>`_.

System Requirements
-------------------
- Python 3.10
- Dependencies listed in `requirements.txt`
- Tested on Python 3.10

Installation Guide
-------------------
1. Create a conda environment:

   .. code-block:: bash

       conda env create -f environment.yml

2. Activate the environment:

   .. code-block:: bash

       conda activate ss_env

3. Install the package in development mode:

   .. code-block:: bash

       pip install -e .

Demo
----

**Large Dataset:**

If you're working with a large dataset, it's essential to set up an efficient data-loading pipeline to ensure smooth training. We recommend reviewing the store-creation notebooks available in the `scTab repository <https://github.com/theislab/scTab/tree/main/notebooks/store_creation>`_. By following these notebooks, you can create a Merlin datamodule, which the framework can seamlessly read.

**Small Dataset or Single Adata Object:**

For small datasets or a single Adata object, a simple PyTorch dataloader is sufficient (like done in our `multiomics application <https://github.com/theislab/ssl_in_scg/blob/master/self_supervision/data/datamodules.py#L173>`_). However, please note that this approach is not implemented for pretraining, as meaningful pretraining requires larger datasets.

**Adapting Data Path and Running Models:**

Once you have set up the appropriate data-loading pipeline, you can proceed to adapt the data path in the respective training step and run the models.

**Expected output:**

Upon running the models on your data, the expected output will be a checkpoint file containing the trained model parameters. This checkpoint file is saved using PyTorch Lightning's built-in functionality for checkpointing. The checkpoint file serves as a snapshot of the model's state after training and can be used for inference, further training, or reproducibility purposes.

**Expected run time for demo on a "normal" desktop computer:**

Our experiments were performed on GPUs, and thus, our models have not been tested explicitly on CPUs. For optimal performance, we recommend running the pre-training phase on a single GPU for approximately 1-2 days. Subsequently, for fine-tuning, we suggest allocating approximately 12-24 hours on a single GPU for the MLPs deployed in our study.

Instructions for Use
--------------------

Obtain the dataset from the scTab repository: [Insert link to scTab repository here]
Follow the instructions provided in the README of this repository to run the models on your data.

Licence
-------
`self_supervision` is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

Authors
-------

`ssl_in_scg` was written by `Till Richter <till.richter@helmholtz-muenchen.de>`_, `Mojtaba Bahrami <mojtaba.bahrami@helmholtz-muenchen.de>`_, `Yufan Xia <yufan.xia@helmholtz-muenchen.de>`_ and `Felix Fischer  <felix.fischer@helmholtz-muenchen.de>`_ .
