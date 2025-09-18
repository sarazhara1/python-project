"""
Dataset loading utilities for clustering demonstrations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris as sk_load_iris, make_blobs
from typing import Tuple, Optional

def load_iris() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load the Iris dataset for clustering demonstrations.
    
    Returns:
        Tuple containing:
        - X: Feature matrix (150, 4)
        - y: True labels for evaluation (150,)
        - feature_names: List of feature names
    """
    iris = sk_load_iris()
    return iris.data, iris.target, iris.feature_names

def load_blobs(n_samples: int = 300, centers: int = 4, 
               cluster_std: float = 1.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
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
        - y: True cluster labels
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
                      cluster_std=cluster_std, random_state=random_state)
    return X, y

def load_customer_data() -> Optional[pd.DataFrame]:
    """
    Load customer segmentation data (placeholder for CSV loading).
    
    Returns:
        DataFrame with customer features, or None if file not found
    """
    try:
        # Placeholder - in real implementation, would load from CSV
        # return pd.read_csv('customer_data.csv')
        
        # Generate synthetic customer data for demonstration
        np.random.seed(42)
        n_customers = 200
        
        age = np.random.normal(35, 12, n_customers)
        income = np.random.normal(50000, 20000, n_customers)
        spending = np.random.normal(2000, 800, n_customers)
        
        df = pd.DataFrame({
            'age': age,
            'annual_income': income,
            'spending_score': spending
        })
        
        return df
    except FileNotFoundError:
        print("Customer data file not found. Using synthetic data.")
        return None

