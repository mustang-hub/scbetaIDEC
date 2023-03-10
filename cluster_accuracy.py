import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T
    acc = sum([w[i,j] for i,j in ind ]) * 1.0 / y_pred.size
    return acc

