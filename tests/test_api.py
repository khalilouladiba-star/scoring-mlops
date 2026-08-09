"""Tests basiques de l'API, utilisés dans la pipeline CI/CD."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

# Le context manager déclenche les événements startup/shutdown de FastAPI
# (nécessaire pour que le modèle soit chargé avant les tests).
with TestClient(app) as client:

    def test_health():
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_valid_payload():
        payload = {
            "age": 35,
            "monthly_income": 3000,
            "debt_ratio": 0.3,
            "revolving_utilization": 0.3,
            "num_credit_lines": 5,
            "num_late_30_59": 0,
            "num_late_60_89": 0,
            "num_late_90_plus": 0,
            "num_dependents": 1,
            "num_real_estate_loans": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["default_probability"] <= 1
        assert body["prediction"] in (0, 1)

    def test_predict_invalid_payload():
        response = client.post("/predict", json={"age": -5})
        assert response.status_code == 422
