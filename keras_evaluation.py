from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import json

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

import sklearn.metrics
def print_roc_auc(model, y_res, y_test):
    #model.summary()
    #score = model.predict(X_test)
    fpr, tpr, auc_threshold = sklearn.metrics.roc_curve(y_test, y_res, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.show()
    print("ROC AUC: %.1f%%" % (100.0*float(roc_auc)))
    return roc_auc, fpr, tpr

def roc_auc(model, X_test, y_test):
    #model.summary()
    #score = model.predict(X_test)
    y_res = model.predict(X_test)
    fpr, tpr, auc_threshold = sklearn.metrics.roc_curve(y_test, y_res, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
#    plt.plot(fpr, tpr)
#    plt.show()
#    print("ROC AUC: %.1f%%" % (100.0*float(roc_auc)))
    return roc_auc, fpr, tpr

def confusion(y_res, y_test, threshold=0.5):
    res = Counter(list(zip([1*x[0] for x in (y_res > threshold)], y_test)))
    return np.array([[res[(0,0)], res[(1,0)]], [res[(0,1)],res[(1,1)]]])

from collections import Counter
# Final evaluation of the model
def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model agains a testing dataset"""
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Scores", json.dumps(scores))
    print("Result samples:")
    y_res = model.predict(X_test)
    print([1*x[0] for x in (y_res > 0.5)][:20])
    print(y_test[:20])
    rocauc, fpr, tpr = print_roc_auc(model, y_res, y_test)
    print(confusion(y_res, y_test, 0.5))
    return rocauc, fpr, tpr


