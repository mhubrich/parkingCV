import numpy as np
import sklearn.metrics


def log_loss(y_true, y_pred):
    return sklearn.metrics.log_loss(y_true, y_pred)


def accuracy_score(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, np.round(y_pred))


def roc_auc_score(y_true, y_pred):
    return sklearn.metrics.roc_auc_score(y_true, y_pred)


def confusion_matrix(y_true, y_pred):
    return sklearn.metrics.confusion_matrix(y_true, np.round(y_pred))


def TP(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    return float(conf[1,1]) / (conf[1,1] + conf[1,0])


def TN(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    return float(conf[0,0]) / (conf[0,0] + conf[0,1])
