import numpy as np
import faiss


def perform_knn(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Perform k-Nearest Neighbors classification using train embeddings to classify test embeddings.

    Args:
        reference_embeddings (numpy.ndarray): The embeddings of the reference set.
        reference_labels (numpy.ndarray): The corresponding labels for each reference embedding.
        test_embeddings (numpy.ndarray): The embeddings of the test set to classify.
        k (int): Number of nearest neighbors to use for classification.
        use_gpu (bool): Whether to use GPU resources for Faiss index.

    Returns:
        numpy.ndarray: The predicted labels for each test embedding.
    """
    print("Start kNN...")
    # Ensure that the embeddings and labels are in float32 for FAISS
    reference_embeddings = reference_embeddings.astype(np.float32)
    test_embeddings = test_embeddings.astype(np.float32)

    # Initialize the index
    d = reference_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)

    if use_gpu:
        # Initialize GPU resources and transfer the index to GPU
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add reference embeddings to the index
    index.add(reference_embeddings)

    # Perform the search for the k-nearest neighbors
    # Note: We search for `k + 1` neighbors because the closest point is expected to be the point itself
    D, I = index.search(test_embeddings, k + 1)

    # Process the results:
    # Exclude the first column of indices (self-match)
    # and use the remaining columns to vote for the predicted label
    y_pred = np.array(
        [np.argmax(np.bincount(reference_labels[I[i, 1:]])) for i in range(I.shape[0])]
    )

    return y_pred
