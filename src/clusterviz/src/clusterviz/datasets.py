from sklearn.datasets import load_iris as sk_load_iris, make_blobs
import numpy as np
import pandas as pd
from typing import Tuple

def load_iris() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load the classic Iris dataset.

    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    """
    iris = sk_load_iris()
    return iris.data, iris.target, iris.feature_names.tolist()

def load_blobs(n_samples: int = 300, centers: int = 4,
               cluster_std: float = 1.0, random_state: int = 42) -> Tuple[np.ndarray, 
np.ndarray]:
    """
    Generate synthetic blob clusters for testing clustering algorithms.

    Args:
        n_samples: Number of samples to generate
        centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
        - X: Generated feature matrix
        - y: Cluster labels
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers,
                      cluster_std=cluster_std, random_state=random_state)
    return X, y

def load_customer_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data for clustering.

    Args:
        n_samples: Number of customers to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame containing synthetic customer data
    """
    rng = np.random.default_rng(random_state)
    data = {
        "age": rng.integers(18, 70, n_samples),
        "income": rng.normal(50000, 15000, n_samples).astype(int),
        "spending_score": rng.integers(1, 100, n_samples)
    }
    return pd.DataFrame(data)

