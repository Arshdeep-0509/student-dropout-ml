"""
Shared helpers used by BOTH the training notebook (model/train_models.ipynb)
and the Streamlit app (app.py).

Keeping the metric code in one place guarantees that the numbers printed in the
notebook (and copied into the README) are computed exactly the same way as the
numbers shown in the web app.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Order in which the six required metrics are displayed everywhere
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def clean_columns(df):
    """Strip stray whitespace / BOM characters from column names.

    The original UCI file has a tab at the end of 'Daytime/evening attendance'
    and a BOM before the first column, which would break column matching.
    """
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def evaluate(model, X, y_true):
    """Score a fitted classifier / pipeline on (X, y_true).

    Returns
    -------
    metrics : dict   {metric name: value} for the 6 required metrics
    y_pred  : array  predicted class indices (0..n-1)
    y_proba : array  predicted class probabilities, shape (n_samples, n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    classes = model.classes_          # e.g. [0, 1, 2]
    n_classes = len(classes)

    # Binary problems use the "positive class" metrics,
    # multi-class problems average the per-class scores weighted by class size.
    average = "binary" if n_classes == 2 else "weighted"

    # AUC needs probabilities, not hard predictions.
    # Wrapped in try/except because AUC is undefined when the uploaded file
    # contains only one class (e.g. a user uploads a tiny subset).
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            # One-vs-Rest: one AUC per class, then a weighted average
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted", labels=classes
            )
    except ValueError:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        # zero_division=0 avoids warnings when a class is never predicted
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred, y_proba
