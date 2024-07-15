import argparse
import scvi
import scanpy as sc
import os
import numpy as np
from self_supervision.paths import RESULTS_FOLDER


def train_scvi(adata, key='scVI'):
    """
    Train the scVI model
    :param adata: anndata object
    :return:
    """
    print('Load adata...')
    print('Setup scVI model...')

    X_normalized = adata.X
    adata.X = np.expm1(adata.X)
    scvi.model.SCVI.setup_anndata(adata, batch_key='dataset_id')

    vae = scvi.model.SCVI(adata, gene_likelihood='nb', n_layers=5, n_latent=64)
    print('Train scVI model...')
    vae.train()
    # print('Save scVI model...')
    # vae.save(os.path.join(model_dir, 'final_models', 'reconstruction'))
    
    adata.obsm[key] = vae.get_latent_representation()
    adata.X = X_normalized

    return adata


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_file', type=str, default='adata_test_embs_scib.h5ad')
        parser.add_argument('--output_file', type=str, default='adata_test_embs_scib.h5ad')
        args = parser.parse_args()
        return args
    
    args = parse_args()

    EMB_RESULT_FOLDER = os.path.join(RESULTS_FOLDER, 'embedding')

    adata = sc.read_h5ad(os.path.join(EMB_RESULT_FOLDER, args.input_file))
    adata = train_scvi(adata)
    adata.write(os.path.join(EMB_RESULT_FOLDER, args.output_file))
