import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import SEED
from src.costs import expected_cost

COST_THRESHOLD = 1 / 6  # refuse when P(bad) is above this, from the 5:1 matrix


def cross_validate(model, X, y, n_splits=5, n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=SEED)
    rows = []
    for train, test in cv.split(X, y):
        m = clone(model)                      # fresh model every fold
        m.fit(X.iloc[train], y.iloc[train])   # preprocessing is fitted here only
        p = m.predict_proba(X.iloc[test])[:, 1]
        yt = y.iloc[test]
        p50 = (p >= 0.5).astype(int)
        pc = (p >= COST_THRESHOLD).astype(int)
        rows.append({
            "roc_auc": roc_auc_score(yt, p),
            "brier": brier_score_loss(yt, p),
            "precision": precision_score(yt, p50, zero_division=0),
            "recall": recall_score(yt, p50),
            "f1": f1_score(yt, p50),
            "cost_at_0.5": expected_cost(yt, p50),
            "cost_at_1/6": expected_cost(yt, pc),
        })
    return pd.DataFrame(rows)