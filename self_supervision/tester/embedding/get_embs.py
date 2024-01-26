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
from self_supervision.trainer.reconstruction.train import update_weights
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default='/lustre/groups/ml01/workspace/till.richter/merlin_cxg_2023_05_15_sf-log1p', type=str,
                        help='Path to the data stored as parquet files')
    parser.add_argument('--model_path', default='/lustre/groups/ml01/workspace/till.richter/trained_models/final_models', type=str,
                        help='Path where the lightning checkpoints are stored')
    parser.add_argument('--result_path', default='/lustre/groups/ml01/workspace/mojtaba.bahrami/ssl_results/', type=str,
                        help='Path to save the output adata object containing the embeddings over the test set')
    parser.add_argument('--hvg', default=False, type=bool,
                        help='Whether to use highly variable genes for training')
    parser.add_argument('--split',
                        default='test', type=str,
                        help='The split to use for evaluation.')
    parser.add_argument('--supervised_subset', default=None, type=str,
                        choices=[None, 'PBMCs_Integration', 'Lung_Integration'],
                        help='Dataset name')

    return parser.parse_args()


args = parse_args()
print(args)

HVG = args.hvg
MODEL_PATH = args.model_path
split = args.split

print("Loading test data...")
adata = sc.read_h5ad(
    f"/lustre/groups/ml01/workspace/mojtaba.bahrami/cellxgene/cellxgene_{split}_adata.h5ad"
)

model_dirs = {
    ('reconstruction', 'Supervised-0'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_No_SSL_CN_integration_run0MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'Supervised-1'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_No_SSL_CN_integration_run1MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'Supervised-2'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_No_SSL_CN_integration_run2MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'Supervised-3'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_No_SSL_CN_integration_run3MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'Supervised-4'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_No_SSL_CN_integration_run4MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',

    ('reconstruction', 'SSL-0'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run0MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-1'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run1MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-2'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run2MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-3'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run3MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-4'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_run4MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',

    ('reconstruction', 'SSL-Shallow-0'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run0MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-Shallow-1'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run1MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-Shallow-2'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run2MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-Shallow-3'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run3MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    ('reconstruction', 'SSL-Shallow-4'): '/lustre/groups/ml01/workspace/mojtaba.bahrami/trained_models/final_models/reconstruction/CN_SSL_CN_CN_MLP_50pintegration_shallow_earlystopping_run4MLP__Lung_Integration/default/version_0/checkpoints/best_checkpoint_val.ckpt',
    
    }

# init estim class
estim = EstimatorAutoEncoder(args.data_path, hvg=HVG)

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
adata_train = sc.read_h5ad('/lustre/groups/ml01/workspace/till.richter/log1p_cellxgene_train_adata.h5ad', backed='r')
random_mask = np.random.choice(adata_train.shape[0], 100_000, replace=False)
adata_train = adata_train[random_mask, :]
pca.fit(adata_train.X)
adata.obsm['PCA'] = pca.transform(adata.X.toarray())

adata.write(os.path.join(args.result_path, f'adata_{split}_embs_scib.h5ad'))
