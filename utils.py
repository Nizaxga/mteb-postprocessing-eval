import numpy as np
from sklearn.decomposition import PCA


def embedding_to_proj(embeddings, origin, directions):
    """HELPER"""

    X = embeddings - origin
    return X @ directions.T / (np.linalg.norm(directions, axis=1) ** 2)


def low_rank_approximation(X, k):
    """HELPER"""

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def all_but_the_top(X, D=5):
    """HELPER"""

    X = X - X.mean(axis=0)
    return X - low_rank_approximation(X, D)


def PCAP(X, D=5):
    """HELPER"""
    X -= np.mean(X, axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n_components = min(128, X.shape[0], X.shape[1])
    # print(f"[LOG] Fitting in to {n_components} dim")
    X = PCA(n_components=n_components).fit_transform(X)
    # return all_but_the_top(X, D)
    return X
