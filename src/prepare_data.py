"""
Étape 2 — Préparation et traitement des données.

- Chargement des données brutes
- Traitement des valeurs manquantes
- Suppression / capping des outliers
- Feature engineering
- Split train/test
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "credit_data.csv"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Valeurs manquantes : imputation par la médiane pour les variables numériques
    for col in df.columns:
        if df[col].isna().any() and col != "target":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 2. Outliers : capping par winsorisation (1er / 99e percentile) sur les
    #    variables continues sensibles aux valeurs extrêmes
    for col in ["monthly_income", "debt_ratio", "revolving_utilization"]:
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower, upper)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total des incidents de paiement en retard
    df["total_late_payments"] = (
        df["num_late_30_59"] + df["num_late_60_89"] + df["num_late_90_plus"]
    )

    # Ratio d'endettement ajusté par le revenu (évite division par zéro)
    df["income_per_dependent"] = df["monthly_income"] / (df["num_dependents"] + 1)

    # Indicateur binaire : a déjà eu un retard sévère (60j+)
    df["has_severe_late_payment"] = (
        (df["num_late_60_89"] > 0) | (df["num_late_90_plus"] > 0)
    ).astype(int)

    # Ratio crédits immobiliers / total lignes de crédit
    df["real_estate_ratio"] = df["num_real_estate_loans"] / (df["num_credit_lines"] + 1)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def run_pipeline(save: bool = True):
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(target=y_train).to_csv(PROCESSED_DIR / "train.csv", index=False)
        X_test.assign(target=y_test).to_csv(PROCESSED_DIR / "test.csv", index=False)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Taux de défaut train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()
