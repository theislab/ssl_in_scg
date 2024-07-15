import argparse
import os
import scanpy as sc
import numpy as np
import lightning.pytorch as pl
from self_supervision.models.lightning_modules.cellnet_autoencoder import (
    MLPAutoEncoder,
    MLPClassifier,
)
from self_supervision.estimator.cellnet import EstimatorAutoEncoder
from sklearn.decomposition import PCA
from self_supervision.paths import DATA_DIR, RESULTS_FOLDER, TRAINING_FOLDER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',
                        default='test', type=str,
                        help='The split to use for evaluation.')
    parser.add_argument('--supervised_subset', default=None, type=str,
                        choices=[None, 'PBMCs_Integration', 'Lung_Integration'],
                        help='Dataset name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    MODEL_PATH = os.path.join(TRAINING_FOLDER, 'final_models')
    split = args.split

    print("Loading test data...")
    adata = sc.read_h5ad(
        f"{DATA_DIR}/merlin_cxg_2023_05_15_sf-log1p/cellxgene_{split}_adata.h5ad"
    )

    base_dir = os.path.join(TRAINING_FOLDER, 'integration')
    model_dirs = {
        ('reconstruction', 'Supervised-0'): os.path.join(base_dir, 'CN_No_SSL_CN_integration_run0MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'Supervised-1'): os.path.join(base_dir, 'reconstruction/CN_No_SSL_CN_integration_run1MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'Supervised-2'): os.path.join(base_dir, 'reconstruction/CN_No_SSL_CN_integration_run2MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'Supervised-3'): os.path.join(base_dir, 'reconstruction/CN_No_SSL_CN_integration_run3MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'Supervised-4'): os.path.join(base_dir, 'reconstruction/CN_No_SSL_CN_integration_run4MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),

        ('reconstruction', 'SSL-0'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run0MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-1'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run1MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-2'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run2MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-3'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run3MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-4'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run4MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),

        ('reconstruction', 'SSL-Shallow-0'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run0MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-Shallow-1'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run1MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-Shallow-2'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run2MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-Shallow-3'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run3MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        ('reconstruction', 'SSL-Shallow-4'): os.path.join(base_dir, 'reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run4MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt'),
        
        }

    # init estim class
    estim = EstimatorAutoEncoder(os.path.join(DATA_DIR, "merlin_cxg_2023_05_15_sf-log1p"))

    # init datamodule
    estim.init_datamodule(batch_size=8192) 

    if split == "test":
        dataloader = estim.datamodule.test_dataloader()
    elif split == "val":
        dataloader = estim.datamodule.val_dataloader()


    print("Predicting embeddings...")
    for (task, name), model_dir in model_dirs.items():
        print("Evaluating model: ", task, name)
        if name in adata.obsm.keys():
            print("Skipping model: ", task, name)
            continue
        # Load model checkpoint
        if task in ["reconstruction", "masking", "contrastive"]:
            try:
                estim.model = MLPAutoEncoder.load_from_checkpoint(
                    model_dir,
                    **estim.get_fixed_autoencoder_params(),
                    units_encoder=[512, 512, 256, 256, 64],
                    units_decoder=[256, 256, 512, 512],
                    batch_integration=False,
                    batch_integration_dim=3,
                    batch_integration_map=None,
                )
            except Exception as e:
                print(e)
                estim.model = MLPAutoEncoder.load_from_checkpoint(
                    model_dir,
                    **estim.get_fixed_autoencoder_params(),
                    units_encoder=[512, 512, 256, 256, 64],
                    units_decoder=[256, 256, 512, 512],
                    batch_integration=True,
                    batch_integration_dim=3,
                    batch_integration_map=None,
                )
        elif task == "classification":
            estim.model = MLPClassifier.load_from_checkpoint(
                model_dir,
                **estim.get_fixed_clf_params(),
                units=[512, 512, 256, 256, 64],
                batch_integration=True,
                batch_integration_dim=3,
                batch_integration_map=None,
            )
        else:
            raise ValueError(
                f'task has to be in ["reconstruction", "classification", "masking", "contrastive"]. You supplied: {task}'
            )

        estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

        estim.model.predict_embedding = True
        emb_preds = estim.predict_embedding(dataloader)
        adata.obsm[name] = emb_preds


    split = args.split
    if args.supervised_subset == 'PBMCs_Integration':
        dataset_ids = [243, 179, 225]
    elif args.supervised_subset == 'Lung_Integration':
        dataset_ids = [131, 59, 205]
    else:
        dataset_ids = None

    adata = adata[adata.obs['dataset_id'].isin(dataset_ids), :]
    adata.obs_names_make_unique()


    # run pca by scikit-learn
    pca = PCA(n_components=64)
    adata_train = sc.read_h5ad(os.path.join(DATA_DIR, 'log1p_cellxgene_train_adata.h5ad'), backed='r')
    random_mask = np.random.choice(adata_train.shape[0], 100_000, replace=False)
    adata_train = adata_train[random_mask, :]
    pca.fit(adata_train.X)
    adata.obsm['PCA'] = pca.transform(adata.X.toarray())

    adata.write(os.path.join(RESULTS_FOLDER, f'adata_{split}_embs_scib.h5ad'))
