import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """Visualization utilities for 1D numeric data."""
    def __init__(self, data):
        self.data = data

    def histogram(self, bins=10):
        sns.histplot(self.data, bins=bins, kde=True)
        plt.xlabel("Value"); plt.ylabel("Frequency"); plt.title("Histogram")
        plt.tight_layout(); plt.show()

    def line(self):
        plt.plot(self.data, marker="o")
        plt.xlabel("Index"); plt.ylabel("Value"); plt.title("Line Plot")
        plt.tight_layout(); plt.show()
