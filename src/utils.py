# File: src/utils.py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def timer(func):
    """Decorator to time a function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result
    return wrapper


def save_fig(fig, name):
    """Save matplotlib figure."""
    path = f"figures/{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_table(df, name):
    """Save pandas DataFrame as CSV."""
    path = f"tables/{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


# ---------- pure-numpy metric helpers ----------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1):
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
