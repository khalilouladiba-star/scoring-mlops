"""
Adapte le fichier Kaggle "Give Me Some Credit" (cs-training.csv) au format
attendu par le reste du projet (data/raw/credit_data.csv).

Usage:
    python data/adapt_kaggle_data.py /chemin/vers/cs-training.csv
"""
import sys
from pathlib import Path

import pandas as pd

COLUMN_MAPPING = {
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "num_late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "num_credit_lines",
    "NumberOfTimes90DaysLate": "num_late_90_plus",
    "NumberRealEstateLoansOrLines": "num_real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "num_late_60_89",
    "NumberOfDependents": "num_dependents",
}


def adapt(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path, index_col=0)
    df = df.rename(columns=COLUMN_MAPPING)
    df = df[list(COLUMN_MAPPING.values())]
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Fichier adapté sauvegardé : {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Taux de défaut: {df['target'].mean():.2%}")


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cs-training.csv")
    output_path = Path(__file__).parent / "raw" / "credit_data.csv"
    adapt(input_path, output_path)
