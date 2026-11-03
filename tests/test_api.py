from fastapi.testclient import TestClient

from gamma_lab.app import app


def test_rejects_bad_features():
    client = TestClient(app)
    assert client.post("/predict", json={"features": [1, 2]}).status_code == 422
    assert client.post("/predict", json={"features": ["NaN"] * 10}).status_code == 422
