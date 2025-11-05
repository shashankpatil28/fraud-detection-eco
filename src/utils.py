# File: src/utils.py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools # <-- NEW IMPORT


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
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label)) #<-- ADDED TN

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Return confusion matrix components as well
    return precision, recall, f1, (tp, fp, fn, tn)


# -----------------------------------------------------------------
# --- NEW FUNCTION START ---
# -----------------------------------------------------------------
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function creates and saves a confusion matrix plot.
    cm: tuple (tp, fp, fn, tn)
    """
    tp, fp, fn, tn = cm
    matrix = np.array([[tn, fp],
                       [fn, tp]])
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes, rotation=90, va='center')

    # Loop over data dimensions and create text annotations.
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        ax.text(j, i, format(matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    
    # Save the figure
    save_fig(fig, title.lower().replace(' ', '_'))

# -----------------------------------------------------------------
# --- NEW FUNCTION END ---
# -----------------------------------------------------------------