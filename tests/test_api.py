from fastapi.testclient import TestClient

from gamma_lab.app import app


def test_rejects_bad_features():
    client = TestClient(app)
    assert client.post("/predict", json={"features": [1, 2]}).status_code == 422
    assert client.post("/predict", json={"features": ["NaN"] * 10}).status_code == 422
    assert client.post("/predict", json={"features": [1e308] * 10}).status_code == 422


def test_valid_request_uses_packaged_model():
    client = TestClient(app)
    response = client.post("/predict", json={"features": [100, 20, 2, 0.4, 0.2, 0, 0, 0, 20, 100]})
    assert response.status_code == 200
    assert 0 <= response.json()["gamma_score"] <= 1
    assert response.json()["simulation_only"] is True


def test_demo_page_available():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Simulated telescope" in response.text
