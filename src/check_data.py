
from src.data import load_data, NUMERIC, CATEGORICAL

X, y = load_data()
print("Shape of X:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Numeric columns:", len(NUMERIC), "| Categorical columns:", len(CATEGORICAL))
print("Class counts (0 = good, 1 = bad):")
print(y.value_counts().sort_index())
print("Share of bad risks:", round(y.mean(), 3))
