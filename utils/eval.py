import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_recall_fscore_support

def accuracy(preds, labels):
    """Computes accuracy: (correct predictions / total predictions)"""
    preds = np.array(preds)
    labels = np.array(labels)

    correct = (preds == labels.squeeze(1)).sum().item()
    total = labels.size

    return (correct / total) * 100

def sensitivity(preds, labels):
    """Computes sensitivity (recall) for each class"""
    _, recall, _, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return recall

def f1_score(preds, labels):
    """Computes F1-score for each class"""
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return f1

def confusion_matrix(preds, labels):
    """Computes the confusion matrix"""
    return sk_confusion_matrix(labels, preds)
