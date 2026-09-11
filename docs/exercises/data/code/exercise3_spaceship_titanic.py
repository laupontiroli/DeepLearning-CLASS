"""Exercise 3 — Spaceship Titanic preprocessing.

Carrega o dataset real, faz split estratificado, aplica imputação, encoding e scaling,
com o objetivo de preparar os dados para uma rede com tanh.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def describe_data(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Target balance:")
    print(df["Transported"].value_counts(normalize=True))
    print("\nMissing values:")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss)
    print("\nMissing ratios:")
    print((df.isna().mean() * 100).sort_values(ascending=False).round(2))

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    desc = df[spend_cols].agg(["mean", "median", "max"])
    print("\nSpending summary:")
    print(desc)


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    id_cols = ["PassengerId", "Cabin", "Name"]

    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Missing values: numerical -> median, categorical -> most frequent
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_cols = [c for c in X_train.columns if c not in categorical_cols + id_cols and c not in spend_cols]
    cat_cols = categorical_cols

    # keep all input columns except identifiers dropped for model input
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=id_cols)

    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])

    X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat = cat_imputer.transform(X_test[cat_cols])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_ohe = ohe.fit_transform(X_train_cat)
    X_test_cat_ohe = ohe.transform(X_test_cat)

    X_train_spend = X_train[spend_cols].copy()
    X_test_spend = X_test[spend_cols].copy()

    # apply log1p to heavy-tailed spending columns
    for col in spend_cols:
        X_train_spend[col] = np.log1p(X_train_spend[col].fillna(0.0))
        X_test_spend[col] = np.log1p(X_test_spend[col].fillna(0.0))

    # combine features
    X_train_proc = np.hstack([X_train_num, X_train_cat_ohe, X_train_spend.to_numpy()])
    X_test_proc = np.hstack([X_test_num, X_test_cat_ohe, X_test_spend.to_numpy()])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    info = {
        "train_shape": X_train_scaled.shape,
        "test_shape": X_test_scaled.shape,
        "min_train": X_train_scaled.min(),
        "max_train": X_train_scaled.max(),
        "min_test": X_test_scaled.min(),
        "max_test": X_test_scaled.max(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "spend_cols": spend_cols,
        "ohe": ohe,
        "scaler": scaler,
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, info


def save_figure_6(df: pd.DataFrame) -> None:
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int).to_numpy()
    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    raw_foodcourt = X_train["FoodCourt"].dropna()
    transformed_foodcourt = np.log1p(raw_foodcourt.fillna(0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(raw_foodcourt, bins=40, color="steelblue", edgecolor="black")
    axes[0].set_title("FoodCourt — antes do preprocessing")
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("contagem")

    axes[1].hist(transformed_foodcourt, bins=40, color="darkorange", edgecolor="black")
    axes[1].set_title("FoodCourt — depois do log1p")
    axes[1].set_xlabel("log1p(FoodCourt)")
    axes[1].set_ylabel("contagem")

    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "figures" / "fig06-foodcourt-preprocessing.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    describe_data(df)
    save_figure_6(df)

    X_train, X_test, y_train, y_test, info = preprocess(df)

    print("\nFinal training shape:", X_train.shape)
    print("Final test shape:", X_test.shape)
    print("NaN remaining in train:", np.isnan(X_train).sum())
    print("NaN remaining in test:", np.isnan(X_test).sum())
    print("Scaled range train:", X_train.min(), X_train.max())
    print("Scaled range test:", X_test.min(), X_test.max())


if __name__ == "__main__":
    main()
