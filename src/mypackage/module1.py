import numpy as np
import pandas as pd

class DataAnalyzer:
    """Basic analysis utilities for numeric sequences."""
    def __init__(self, data):
        self.data = np.asarray(pd.Series(data).dropna(), dtype=float)

    def mean(self):
        return float(np.mean(self.data))

    def variance(self, ddof=0):
        return float(np.var(self.data, ddof=ddof))

    def describe(self):
        return {
            "count": int(self.data.size),
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data, ddof=1)) if self.data.size > 1 else 0.0,
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
        }

    def moving_average(self, window=3):
        if window <= 0 or window > self.data.size:
            raise ValueError("window must be in 1..len(data)")
        return np.convolve(self.data, np.ones(window)/window, mode="valid")
