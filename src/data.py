
import pandas as pd

from src.config import RAW_DATA

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "people_liable", "telephone", "foreign_worker", "class",
]

NUMERIC = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "people_liable",
]

CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]


def load_raw():
    """Read german.data exactly as provided (no header, space separated)."""
    return pd.read_csv(RAW_DATA, sep=r"\s+", header=None, names=COLUMNS)


def load_data():
    """Return X (20 attributes) and y (1 = bad credit risk, 0 = good)."""
    df = load_raw()
    y = (df["class"] == 2).astype(int)
    X = df.drop(columns="class")
    return X, y
