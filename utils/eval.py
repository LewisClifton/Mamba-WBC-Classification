import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_recall_fscore_support

def accuracy(preds, labels):
    """Computes accuracy: (correct predictions / total predictions)"""
    preds = np.array(preds)
    labels = np.array(labels)
    return np.mean(preds == labels)

def sensitivity(preds, labels):
    """Computes sensitivity (recall) for each class"""
    _, recall, _, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return recall  # Returns an array with recall for each class

def f1_score(preds, labels):
    """Computes F1-score for each class"""
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return f1  # Returns an array with F1-score for each class

def confusion_matrix(preds, labels):
    """Computes the confusion matrix"""
    return sk_confusion_matrix(labels, preds)
