import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_recall_fscore_support

def get_eval_metrics(preds, labels):
    """Gets accuracy, precision, sensitivity, F1-score, and confusion matrix."""
    preds = np.array(preds)
    labels = np.array(labels)

    # Accuracy
    correct = (preds == labels.squeeze(1)).sum().item()
    total = labels.size
    accuracy = (correct / total) * 100

    # Precision, recall, F1-score
    precision, sensitivity, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    # Confusion matrix
    conf_matrix = sk_confusion_matrix(labels, preds)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "F1 Score": f1,
        "Confusion Matrix": conf_matrix
    }
