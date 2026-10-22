import pandas as pd
from app.main import app
from app.naive import NaiveForecaster
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()
    assert isinstance(r.json()["model_loaded"], bool)


def test_predict_structure():
    r = client.get("/predict", params={"store": "store_1", "days": 3})
    # Модель может не быть обучена в тестовом окружении — допускаем 503
    assert r.status_code in (200, 503, 404)
    if r.status_code == 200:
        body = r.json()
        assert "store" in body
        assert "predictions" in body
        assert len(body["predictions"]) == 3
        for row in body["predictions"]:
            assert {"date", "predicted_sales", "predicted_surplus"} <= row.keys()
            assert row["predicted_sales"] >= 0
            assert row["predicted_surplus"] >= 0


def test_predict_requires_store():
    r = client.get("/predict", params={"days": 3})
    assert r.status_code == 422


def test_predict_validates_days():
    r = client.get("/predict", params={"store": "store_1", "days": 0})
    assert r.status_code == 422


def test_naive_forecaster_fallback():
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=28, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 4,
        }
    )
    model = NaiveForecaster.fit(history)
    future = model.make_future_dataframe(periods=3)
    assert len(future) == 3
    predictions = model.predict(future)
    assert len(predictions) == 3
    assert predictions["yhat"].notna().all()
