import numpy as np
from sklearn.metrics import f1_score

def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro')

def root_mean_squared_logarithmic_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))