"""
Étape 3 — Définition du score métier.

Contexte métier (scoring de crédit) :
- Un FAUX NÉGATIF (le modèle prédit "pas de défaut" alors que le client fait
  réellement défaut) coûte cher à la banque : perte du capital prêté, frais
  de recouvrement, etc. -> coût élevé.
- Un FAUX POSITIF (le modèle prédit "défaut" alors que le client aurait
  remboursé) coûte le manque à gagner sur les intérêts + le risque de perdre
  le client au profit d'un concurrent -> coût plus faible que le FN, mais
  non négligeable.

On pose ici, à titre d'hypothèse métier (à documenter/justifier en
soutenance) :
    coût(FN) = 10   (perte de capital)
    coût(FP) = 1    (manque à gagner commercial)

Le "score métier" est un coût total normalisé, à MINIMISER. On fournit
aussi une fonction de "business gain" à MAXIMISER pour l'utiliser comme
scorer sklearn/MLflow.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

COST_FALSE_NEGATIVE = 10  # rater un vrai défaut
COST_FALSE_POSITIVE = 1  # refuser à tort un bon client


def business_cost(y_true, y_pred, cost_fn: float = COST_FALSE_NEGATIVE, cost_fp: float = COST_FALSE_POSITIVE) -> float:
    """Coût métier total normalisé par le nombre d'observations (à minimiser)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = cost_fn * fn + cost_fp * fp
    return total_cost / len(y_true)


def business_gain(y_true, y_pred, cost_fn: float = COST_FALSE_NEGATIVE, cost_fp: float = COST_FALSE_POSITIVE) -> float:
    """Score métier à MAXIMISER : gain relatif par rapport au pire cas possible.

    On normalise entre 0 (pire cas : tout faux) et 1 (meilleur cas : tout juste),
    ce qui le rend directement comparable à d'autres métriques comme l'AUC.
    """
    worst_cost = cost_fn * y_true.sum() + cost_fp * (len(y_true) - y_true.sum())
    cost = business_cost(y_true, y_pred, cost_fn, cost_fp) * len(y_true)
    if worst_cost == 0:
        return 1.0
    return 1 - (cost / worst_cost)


def find_optimal_threshold(y_true, y_proba, cost_fn: float = COST_FALSE_NEGATIVE, cost_fp: float = COST_FALSE_POSITIVE):
    """Cherche le seuil de décision qui minimise le coût métier (plutôt que 0.5 par défaut)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        costs.append(business_cost(y_true, y_pred, cost_fn, cost_fp))
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


# Scorer utilisable directement dans GridSearchCV(scoring=business_scorer)
business_scorer = make_scorer(business_gain, greater_is_better=True)
