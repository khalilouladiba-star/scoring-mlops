"""
Étape 7 — Analyse du data drift et stratégie de surveillance.

Deux approches complémentaires sont proposées :

1. Un calcul manuel du PSI (Population Stability Index) par variable,
   simple à comprendre et à intégrer dans une alerte automatisée
   (ex: cron job, tâche planifiée dans le pipeline CI/CD).

2. Un rapport détaillé avec la librairie "evidently", qui produit un
   rapport HTML complet (distribution des features, drift des targets,
   corrélations) — pratique pour la soutenance et pour un monitoring
   plus riche.

Indicateurs recommandés à monitorer en production :
  - PSI par feature (seuils usuels : <0.1 stable, 0.1-0.25 drift modéré,
    >0.25 drift fort -> alerte)
  - Taux de prédictions positives du modèle dans le temps (proxy simple
    et peu coûteux, calculable même sans les vraies labels)
  - Performance réelle (AUC, coût métier) dès que les vrais labels sont
    disponibles (ex: après la période de remboursement)
  - Distribution des probabilités prédites (détecte un glissement du
    score global, même si aucune feature individuelle ne dérive)

Stratégie d'alerte / réentraînement proposée :
  - PSI > 0.25 sur une feature critique -> alerte automatique (email/Slack)
    à l'équipe MLOps + investigation manuelle
  - Baisse de l'AUC en production de plus de 5 points par rapport à l'AUC
    de référence (mesurée sur un échantillon labellisé a posteriori)
    -> déclenchement automatique d'un pipeline de réentraînement
  - Réentraînement planifié par défaut tous les mois, indépendamment du
    drift détecté, pour absorber les évolutions lentes non détectées
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre une distribution de référence (train)
    et une distribution courante (production)."""
    reference = reference.dropna()
    current = current.dropna()

    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    # éviter les divisions par zéro / log(0)
    ref_counts = np.where(ref_counts == 0, 1e-6, ref_counts)
    cur_counts = np.where(cur_counts == 0, 1e-6, cur_counts)

    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
    return float(psi)


def psi_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reference_df.columns:
        if col == "target" or not pd.api.types.is_numeric_dtype(reference_df[col]):
            continue
        psi = compute_psi(reference_df[col], current_df[col])
        status = "🟢 stable" if psi < 0.1 else ("🟠 drift modéré" if psi < 0.25 else "🔴 drift fort")
        rows.append({"feature": col, "psi": round(psi, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def simulate_drifted_data(reference_df: pd.DataFrame, drift_strength: float = 0.4) -> pd.DataFrame:
    """Simule des données de production avec un drift injecté (pour la démo).
    En production réelle, `current_df` serait simplement le flux de nouvelles
    requêtes reçues par l'API, agrégées sur une fenêtre de temps (ex: 1 semaine)."""
    drifted = reference_df.copy()
    drifted["revolving_utilization"] = drifted["revolving_utilization"] * (1 + drift_strength)
    drifted["monthly_income"] = drifted["monthly_income"] * (1 - drift_strength * 0.3)
    drifted["debt_ratio"] = np.clip(drifted["debt_ratio"] + drift_strength * 0.2, 0, 2)
    return drifted


def generate_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: Path):
    """Génère un rapport HTML détaillé avec evidently (si disponible)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference_df, current_data=current_df)
        result.save_html(str(output_path))
        print(f"Rapport evidently généré : {output_path}")
    except Exception as e:
        print(f"Rapport evidently non généré (version/API différente) : {e}")
        print("Le rapport PSI manuel ci-dessus reste disponible en fallback.")


if __name__ == "__main__":
    reference = pd.read_csv(PROCESSED_DIR / "train.csv")
    current = simulate_drifted_data(reference)  # à remplacer par les vraies données de prod

    report = psi_report(reference, current)
    print("=== Rapport de data drift (PSI) ===")
    print(report.to_string(index=False))

    output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)
    report.to_csv(output_dir / "psi_report.csv", index=False)

    generate_evidently_report(
        reference.drop(columns=["target"]),
        current.drop(columns=["target"]),
        output_dir / "drift_report.html",
    )
