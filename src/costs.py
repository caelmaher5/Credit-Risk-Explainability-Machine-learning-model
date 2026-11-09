import numpy as np

COST_FN = 5  # bad customer predicted good
COST_FP = 1  # good customer predicted bad


def expected_cost(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)