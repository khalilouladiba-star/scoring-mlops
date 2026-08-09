"""
Étape 2 (partie 1) — Génération d'un jeu de données de scoring de crédit.

IMPORTANT :
Ce script génère un jeu de données SYNTHÉTIQUE qui imite la structure du
kernel Kaggle "Give Me Some Credit" (classification binaire : défaut de
paiement / pas de défaut), avec des features réalistes et un déséquilibre
de classes proche de la réalité (~6-7% de défauts).

Dans le cadre de l'examen, vous devez :
  1. Aller sur Kaggle et choisir un vrai kernel de classification binaire
     avec du feature engineering (ex: "Give Me Some Credit",
     "Home Credit Default Risk", "Credit Card Fraud Detection"...).
  2. Télécharger le CSV réel et le placer dans data/raw/credit_data.csv
  3. Remplacer l'appel à generate_synthetic_data() par pd.read_csv(...)

Ce script sert de fallback pour que tout le pipeline (MLflow, entraînement,
API, Streamlit, drift) soit exécutable et testable de bout en bout dès
maintenant, sans dépendre d'un téléchargement externe.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42


def generate_synthetic_data(n_samples: int = 15000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 75, n_samples)
    monthly_income = rng.gamma(shape=3.0, scale=1800, size=n_samples).round(0)
    debt_ratio = np.clip(rng.beta(2, 5, n_samples), 0, 1)
    num_credit_lines = rng.integers(0, 20, n_samples)
    num_late_payments_30_59 = rng.poisson(0.4, n_samples)
    num_late_payments_60_89 = rng.poisson(0.15, n_samples)
    num_late_payments_90 = rng.poisson(0.1, n_samples)
    revolving_utilization = np.clip(rng.beta(2, 3, n_samples), 0, 1.3)
    num_dependents = rng.integers(0, 5, n_samples)
    num_real_estate_loans = rng.integers(0, 4, n_samples)
    years_employed = np.clip(rng.normal(8, 6, n_samples), 0, 45).round(1)

    # score latent combinant les facteurs de risque (plus haut = plus risqué)
    latent = (
        0.9 * revolving_utilization
        + 0.7 * debt_ratio
        + 0.5 * num_late_payments_30_59
        + 0.9 * num_late_payments_60_89
        + 1.3 * num_late_payments_90
        - 0.00004 * monthly_income
        - 0.02 * years_employed
        - 0.03 * age
        + 0.05 * num_dependents
        + rng.normal(0, 0.8, n_samples)
    )
    prob_default = 1 / (1 + np.exp(-2 * (latent - np.quantile(latent, 0.96))))
    target = rng.binomial(1, prob_default)

    df = pd.DataFrame(
        {
            "age": age,
            "monthly_income": monthly_income,
            "debt_ratio": debt_ratio,
            "revolving_utilization": revolving_utilization,
            "num_credit_lines": num_credit_lines,
            "num_late_30_59": num_late_payments_30_59,
            "num_late_60_89": num_late_payments_60_89,
            "num_late_90_plus": num_late_payments_90,
            "num_dependents": num_dependents,
            "num_real_estate_loans": num_real_estate_loans,
            "years_employed": years_employed,
            "target": target,  # 1 = défaut de paiement
        }
    )

    # Injecter quelques valeurs manquantes et outliers réalistes
    missing_idx = rng.choice(df.index, size=int(0.03 * n_samples), replace=False)
    df.loc[missing_idx, "monthly_income"] = np.nan
    outlier_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[outlier_idx, "monthly_income"] = df["monthly_income"].max() * rng.uniform(5, 10, 20)

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    out_path = out_dir / "credit_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset généré : {out_path} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    print(f"Taux de défaut : {df['target'].mean():.2%}")
