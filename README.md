# Projet MLOps — Scoring de crédit

Implémentation complète du cycle MLOps pour un modèle de scoring de crédit
(classification binaire : défaut de paiement / pas de défaut), conforme aux
7 étapes de l'examen "MLOps et Déploiement".

## Structure du projet

```
scoring-mlops/
├── data/
│   ├── generate_data.py      # Génère un dataset synthétique (fallback Kaggle)
│   ├── raw/credit_data.csv   # Données brutes
│   └── processed/            # Train / test après préparation
├── src/
│   ├── prepare_data.py       # Étape 2 : nettoyage, feature engineering, split
│   ├── business_score.py     # Étape 3 : score métier (coût FP vs FN)
│   └── train_models.py       # Étapes 1 & 4 : MLflow + entraînement + comparaison
├── api/
│   ├── main.py                # Étape 5 : API FastAPI
│   └── Dockerfile
├── streamlit_app/
│   └── app.py                 # Étape 6 : interface utilisateur
├── monitoring/
│   └── drift_analysis.py      # Étape 7 : data drift (PSI + evidently)
├── tests/
│   └── test_api.py
├── .github/workflows/cicd.yml # Étape 5 : pipeline CI/CD
├── models/                    # Modèle champion sauvegardé
├── artifacts/                 # Graphiques (ROC, feature importance, SHAP, drift)
└── requirements.txt
```

Tout le pipeline a été exécuté de bout en bout pour valider qu'il fonctionne
(voir résultats ci-dessous).

---

## Étape 1 — MLFlow

Le tracking MLflow utilise un backend **SQLite** local (`mlflow.db`), une
alternative légère à un vrai serveur PostgreSQL, mais qui suit le même
principe (backend base de données, requis depuis que le backend "fichier
pur" de MLflow est en maintenance).

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-scoring")
```

**Pour un vrai déploiement cloud**, remplacez cette ligne par :
```python
mlflow.set_tracking_uri("postgresql://user:password@host:5432/mlflow")
# ou, si vous avez un serveur MLflow distant :
mlflow.set_tracking_uri("http://<votre-serveur>:5000")
```

Lancer l'interface MLflow pour visualiser les runs :
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Puis ouvrir http://localhost:5000. Vous y verrez les paramètres, métriques,
courbes ROC, feature importance et graphiques SHAP de chaque run, ainsi que
le modèle "credit-scoring-champion" dans le Model Registry.

---

## Étape 2 — Préparation des données

Le projet utilise le vrai dataset Kaggle **"Give Me Some Credit"**
(https://www.kaggle.com/c/GiveMeSomeCredit/data), 150 000 clients, ~6.7% de
défauts de paiement.

```bash
# 1. Téléchargez cs-training.csv depuis Kaggle et placez-le où vous voulez
python data/adapt_kaggle_data.py chemin/vers/cs-training.csv
# -> génère data/raw/credit_data.csv au bon format

python src/prepare_data.py       # nettoyage + feature engineering + split
```

*(`data/generate_data.py` reste disponible comme fallback pour générer un
dataset synthétique si vous voulez tester le pipeline sans passer par
Kaggle.)*

Traitements appliqués :
- Imputation des valeurs manquantes par la médiane (revenu mensuel et
  nombre de personnes à charge, ~20% et ~3% de valeurs manquantes)
- Winsorisation (capping 1%/99%) des variables sensibles aux outliers
- 4 nouvelles features : `total_late_payments`, `income_per_dependent`,
  `has_severe_late_payment`, `real_estate_ratio`
- Split stratifié 80/20 (120 000 / 30 000 lignes)

---

## Étape 3 — Score métier

Hypothèse métier posée dans `src/business_score.py` (à adapter/justifier
selon votre contexte réel en soutenance) :

| Erreur | Signification | Coût relatif |
|---|---|---|
| Faux négatif | On accorde un crédit à un client qui fera défaut | **10** (perte de capital) |
| Faux positif | On refuse un client qui aurait remboursé | **1** (manque à gagner) |

Deux fonctions :
- `business_cost` : coût total à **minimiser**
- `business_gain` : version normalisée (0 à 1) à **maximiser**, utilisée
  comme critère de sélection du meilleur modèle
- `find_optimal_threshold` : cherche le seuil de décision (au lieu de 0.5
  par défaut) qui minimise le coût métier

---

## Étape 4 — Entraînement et comparaison de modèles

```bash
cd src && python train_models.py
```

4 modèles comparés, chacun tracké dans MLflow, entraînés sur les 150 000
clients réels du dataset Kaggle "Give Me Some Credit" :

| Modèle | AUC | Coût métier | Gain métier |
|---|---|---|---|
| Baseline (dummy) | 0.50 | 0.68 | 0.57 |
| Logistic Regression | 0.86 | 0.34 | 0.79 |
| **Random Forest** ⭐ | **0.86** | **0.33** | **0.79** |
| XGBoost | 0.86 | 0.34 | 0.79 |

Les 3 modèles se tiennent de très près (AUC ~0.86). **Random Forest** est
retenu comme champion, avec un gain métier légèrement supérieur et un coût
métier légèrement plus faible que les deux autres.

Pour chaque modèle :
- `GridSearchCV` (3-fold) pour l'optimisation des hyperparamètres
- `SMOTE` (imbalanced-learn) dans un `Pipeline` pour gérer le déséquilibre
  de classes (~7% de défauts)
- Feature importance native + graphique **SHAP** (summary plot)
- Le modèle avec le meilleur **gain métier** (pas seulement l'AUC) est
  sélectionné comme "champion", sauvegardé dans `models/champion_model.pkl`
  et enregistré dans le MLflow Model Registry

---

## Étape 5 — API de scoring (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
# Documentation interactive : http://localhost:8000/docs
```

Endpoints :
- `GET /health` : statut de l'API
- `GET /model-info` : métadonnées du modèle en production
- `POST /predict` : reçoit les données d'un client, renvoie la probabilité
  de défaut et la décision

**Versioning** : le code est prévu pour être versionné avec Git (branche
`main`, commits atomiques par étape — ex: `feat: prepare data`,
`feat: business scoring`, `feat: model training`, `feat: API`, etc.)

**Déploiement cloud** — trois options possibles avec le `Dockerfile` fourni :

- **Render / Heroku** (le plus simple) :
  ```bash
  docker build -t credit-scoring-api -f api/Dockerfile .
  # Puis connecter le repo GitHub à Render/Heroku pour un déploiement auto
  ```
- **AWS** : push de l'image sur ECR, déploiement sur ECS Fargate ou App Runner
- **GCP** : `gcloud run deploy` à partir de l'image Docker (Cloud Run)

**CI/CD** (`.github/workflows/cicd.yml`) :
1. `test` : installe les dépendances, vérifie la présence du modèle, lance
   `pytest tests/`
2. `build-and-push` : construit l'image Docker et la pousse sur Docker Hub
3. `deploy` : déclenche le déploiement (ex: webhook Render) sur push vers `main`

Secrets GitHub à configurer : `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`,
`RENDER_DEPLOY_HOOK_URL` (ou équivalent selon la plateforme choisie).

---

## Étape 6 — Interface Streamlit

```bash
streamlit run streamlit_app/app.py
```

L'utilisateur saisit manuellement les caractéristiques d'un client, l'app
appelle `POST /predict` sur l'API et affiche :
- la probabilité de défaut (jauge + métrique)
- la décision (accepté / refusé) avec code couleur
- le seuil de décision utilisé

Pour pointer vers l'API déployée en production plutôt qu'en local :
```bash
API_URL=https://votre-api-deployee.com streamlit run streamlit_app/app.py
```

---

## Étape 7 — Data drift & préparation à la soutenance

```bash
python monitoring/drift_analysis.py
```

Génère :
- `artifacts/psi_report.csv` : PSI (Population Stability Index) par feature
- `artifacts/drift_report.html` : rapport détaillé (librairie `evidently`)

**Indicateurs à monitorer en production :**
1. PSI par feature (seuils : <0.1 stable, 0.1–0.25 drift modéré, >0.25 fort)
2. Taux de prédictions positives dans le temps (proxy sans besoin des labels)
3. Distribution des probabilités prédites
4. AUC / coût métier réel dès que les vrais labels sont disponibles

**Stratégie d'alerte / réentraînement :**
- PSI > 0.25 sur une feature critique → alerte auto (email/Slack) + investigation
- Baisse d'AUC > 5 points vs référence → réentraînement automatique déclenché
- Réentraînement planifié mensuel par défaut, indépendant du drift détecté

**Plan de soutenance suggéré :**
1. Contexte métier et enjeu du score (coûts FP/FN)
2. Données : source, nettoyage, feature engineering
3. Comparatif des modèles (tableau + courbes ROC + SHAP)
4. Démo live : API (Swagger `/docs`) + interface Streamlit
5. Architecture MLOps (schéma : Git → CI/CD → Docker → Cloud)
6. Monitoring : data drift, stratégie de réentraînement
7. Limites et perspectives (vraies données Kaggle, A/B testing, feedback loop)

---

## Installation rapide

```bash
pip install -r requirements.txt
python data/generate_data.py
python src/prepare_data.py
cd src && python train_models.py && cd ..
uvicorn api.main:app --port 8000 &
streamlit run streamlit_app/app.py
```
