"""
Data preprocessing utilities for clustering.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional

class Preprocessor:
    """
    Handles data preprocessing including scaling and dimensionality reduction.
    """
    
    def __init__(self):
        """Initialize preprocessor with empty scaler and PCA objects."""
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.is_fitted = False
        
    def fit_transform(self, X: np.ndarray, scale: bool = True, 
                     pca_components: Optional[int] = None) -> np.ndarray:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            X: Input feature matrix
            scale: Whether to apply standard scaling
            pca_components: Number of PCA components (None for no PCA)
            
        Returns:
            Transformed feature matrix
        """
        X_processed = X.copy()
        
        # Apply scaling if requested
        if scale:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
        
        # Apply PCA if requested
        if pca_components is not None:
            self.pca = PCA(n_components=pca_components, random_state=42)
            X_processed = self.pca.fit_transform(X_processed)
            
        self.is_fitted = True
        return X_processed
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Transformed feature matrix
            
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        X_processed = X.copy()
        
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
            
        if self.pca is not None:
            X_processed = self.pca.transform(X_processed)
            
        return X_processed
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio from PCA if available.
        
        Returns:
            Array of explained variance ratios or None if PCA not fitted
        """
        if self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None

