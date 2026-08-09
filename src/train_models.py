"""
Étapes 1 et 4 — Entraînement, optimisation et comparaison de modèles,
avec tracking MLflow complet (params, métriques, artefacts, modèles).

Modèles comparés :
  - Baseline : DummyClassifier (stratifié)
  - Logistic Regression
  - Random Forest
  - XGBoost

Pour chaque modèle (hors baseline) :
  - GridSearchCV pour l'optimisation des hyperparamètres
  - Gestion du déséquilibre via SMOTE (imblearn) dans un Pipeline
  - Évaluation : AUC, F1, coût métier (business_score.py), seuil optimal
  - Feature importance (native + SHAP)
  - Logging MLflow : params, métriques, courbe ROC, importance des
    variables, modèle enregistré dans le Model Registry

Le meilleur modèle (selon le score métier) est promu en tant que modèle
"champion" utilisé par l'API (Étape 5).
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from business_score import business_cost, business_gain, find_optimal_threshold

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODELS_DIR = Path(__file__).parent.parent / "models"

# --- Étape 1 : configuration du tracking MLflow ---
# En local : backend SQLite (fichier unique, recommandé par MLflow depuis que
# le backend "filesystem pur" est en maintenance). En production, remplacez
# par un backend PostgreSQL : mlflow.set_tracking_uri("postgresql://user:pwd@host:5432/mlflow")
# ou par l'URI de votre serveur MLflow distant : mlflow.set_tracking_uri("http://<host>:5000")
MLFLOW_DB_PATH = Path(__file__).parent.parent / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("credit-scoring")


def load_processed():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]
    return X_train, X_test, y_train, y_test


def log_roc_curve(y_true, y_proba, run_artifact_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    path = ARTIFACTS_DIR / run_artifact_path
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def log_feature_importance(model, feature_names, run_artifact_path: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order][:15][::-1], importances[order][:15][::-1])
    ax.set_title("Feature importance")
    path = ARTIFACTS_DIR / run_artifact_path
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def log_shap_summary(model, X_sample, run_artifact_path: str):
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(shap_values, X_sample, show=False)
        path = ARTIFACTS_DIR / run_artifact_path
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        print(f"SHAP non calculé pour ce modèle: {e}")
        return None


def run_model(name, estimator, param_grid, X_train, X_test, y_train, y_test, use_smote=True):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("clf", estimator))
    pipeline = ImbPipeline(steps)

    grid_params = {f"clf__{k}": v for k, v in param_grid.items()}

    with mlflow.start_run(run_name=name):
        search = GridSearchCV(pipeline, grid_params, scoring="roc_auc", cv=2, n_jobs=1)
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_

        y_proba = best_pipeline.predict_proba(X_test)[:, 1]

        # Seuil optimal selon le coût métier plutôt que 0.5 par défaut
        best_threshold, min_cost = find_optimal_threshold(y_test.values, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cost = business_cost(y_test.values, y_pred)
        gain = business_gain(y_test.values, y_pred)

        # Logging MLflow : params + métriques
        mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("decision_threshold", round(best_threshold, 2))
        mlflow.log_metrics(
            {
                "auc": auc,
                "f1_score": f1,
                "business_cost": cost,
                "business_gain": gain,
            }
        )

        # Artefacts : ROC curve, feature importance, SHAP
        roc_path = log_roc_curve(y_test, y_proba, f"{name}_roc.png")
        mlflow.log_artifact(str(roc_path))

        fi_path = log_feature_importance(best_pipeline.named_steps["clf"], X_train.columns, f"{name}_feature_importance.png")
        if fi_path:
            mlflow.log_artifact(str(fi_path))

        sample = X_test.sample(min(200, len(X_test)), random_state=42)
        sample_scaled = pd.DataFrame(
            best_pipeline.named_steps["scaler"].transform(sample), columns=sample.columns
        )
        shap_path = log_shap_summary(best_pipeline.named_steps["clf"], sample_scaled, f"{name}_shap_summary.png")
        if shap_path:
            mlflow.log_artifact(str(shap_path))

        # Modèle complet (pipeline scaler+clf) loggé dans MLflow.
        # serialization_format="pickle" : le pipeline imblearn (SMOTE) n'est
        # pas reconnu par le nouveau format sécurisé "skops" de MLflow.
        mlflow.sklearn.log_model(best_pipeline, "model", serialization_format="pickle")

        run_id = mlflow.active_run().info.run_id

        print(f"[{name}] AUC={auc:.3f} F1={f1:.3f} coût_métier={cost:.3f} gain_métier={gain:.3f} seuil={best_threshold:.2f}")

        return {
            "name": name,
            "run_id": run_id,
            "pipeline": best_pipeline,
            "auc": auc,
            "f1": f1,
            "business_cost": cost,
            "business_gain": gain,
            "threshold": best_threshold,
        }


def run_baseline(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="baseline_dummy"):
        clf = DummyClassifier(strategy="stratified", random_state=42)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        auc = roc_auc_score(y_test, y_proba)
        cost = business_cost(y_test.values, y_pred)
        gain = business_gain(y_test.values, y_pred)
        mlflow.log_metrics({"auc": auc, "business_cost": cost, "business_gain": gain})
        print(f"[baseline] AUC={auc:.3f} coût_métier={cost:.3f} gain_métier={gain:.3f}")
        return {"name": "baseline", "auc": auc, "business_cost": cost, "business_gain": gain}


def main():
    X_train, X_test, y_train, y_test = load_processed()

    results = [run_baseline(X_train, X_test, y_train, y_test)]

    results.append(
        run_model(
            "logistic_regression",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"C": [0.01, 0.1, 1, 10]},
            X_train, X_test, y_train, y_test,
        )
    )

    results.append(
        run_model(
            "random_forest",
            RandomForestClassifier(random_state=42, n_jobs=1),
            {"n_estimators": [150], "max_depth": [8, 12]},
            X_train, X_test, y_train, y_test,
        )
    )

    results.append(
        run_model(
            "xgboost",
            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
            {"n_estimators": [150], "max_depth": [4, 6], "learning_rate": [0.1]},
            X_train, X_test, y_train, y_test,
        )
    )

    # Sélection du meilleur modèle selon le GAIN métier (pas seulement l'AUC)
    comparable = [r for r in results if "pipeline" in r]
    best = max(comparable, key=lambda r: r["business_gain"])

    print("\n=== Comparatif des modèles ===")
    print(pd.DataFrame(results).drop(columns=["pipeline"], errors="ignore").to_string(index=False))
    print(f"\nMeilleur modèle (score métier) : {best['name']} (run_id={best['run_id']})")

    # Sauvegarde locale du champion pour l'API (Étape 5)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], MODELS_DIR / "champion_model.pkl")
    with open(MODELS_DIR / "champion_metadata.json", "w") as f:
        json.dump(
            {
                "name": best["name"],
                "run_id": best["run_id"],
                "threshold": best["threshold"],
                "auc": best["auc"],
                "business_gain": best["business_gain"],
                "features": list(X_train.columns),
            },
            f,
            indent=2,
        )
    print(f"Modèle champion sauvegardé dans {MODELS_DIR / 'champion_model.pkl'}")

    # Enregistrement dans le MLflow Model Registry
    model_uri = f"runs:/{best['run_id']}/model"
    try:
        mlflow.register_model(model_uri, "credit-scoring-champion")
        print("Modèle enregistré dans le MLflow Model Registry sous 'credit-scoring-champion'.")
    except Exception as e:
        print(f"Enregistrement dans le registry ignoré (backend fichier limité) : {e}")


if __name__ == "__main__":
    main()
