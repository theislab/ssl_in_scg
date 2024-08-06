import numpy as np
import os
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation
import faiss
from scib_metrics.nearest_neighbors import NeighborsOutput
import argparse
from self_supervision.paths import RESULTS_FOLDER


def cal_scib_metrics(adata):

    def faiss_brute_force_nn(X: np.ndarray, k: int):
        """Gpu brute force nearest neighbor search using faiss."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(X.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(X)
        distances, indices = gpu_index.search(X, k)
        del index
        del gpu_index
        # distances are squared
        return NeighborsOutput(indices=indices, distances=np.sqrt(distances))

    biocons = BioConservation(isolated_labels=False)
    bm = Benchmarker(
        adata,
        batch_key="tech_sample",
        label_key="cell_type",
        embedding_obsm_keys=adata.obsm.keys(),
        pre_integrated_embedding_obsm_key="PCA",
        bio_conservation_metrics=biocons,
        n_jobs=-1,
    )
    bm.prepare(neighbor_computer=faiss_brute_force_nn)
    bm.benchmark()

    df = bm.get_results(min_max_scale=False)
    return df



if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_file', type=str, default='adata_test_embs_scib.h5ad')
        parser.add_argument('--output_file', type=str, default='test_scib_scores.csv')
        return parser.parse_args()

    args = parse_args()

    EMB_RESULT_FOLDER = os.path.join(RESULTS_FOLDER, 'embedding')

    split = args.split
    adata = sc.read_h5ad(os.path.join(EMB_RESULT_FOLDER, args.input_file))
    df = cal_scib_metrics(adata)
    df.to_csv(os.path.join(args.data_dir, args.output_file))
    
