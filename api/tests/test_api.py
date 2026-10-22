from contextlib import contextmanager
from datetime import date

import jwt
import main
from fastapi.testclient import TestClient
from main import JWT_ISSUER, JWT_SECRET, app

client = TestClient(app)


def auth_headers():
    token = client.post("/login", json={"store_id": 1}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_login_returns_token():
    r = client.post("/login", json={"store_id": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] > 0
    payload = jwt.decode(
        body["access_token"],
        JWT_SECRET,
        algorithms=["HS256"],
        issuer=JWT_ISSUER,
    )
    assert payload["store_id"] == 1


def test_login_rejects_negative_store_id():
    r = client.post("/login", json={"store_id": -1})
    assert r.status_code == 422


def test_protected_endpoint_requires_token():
    r = client.get("/stores")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_protected_endpoint_rejects_invalid_token():
    r = client.get("/stores", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_predict_requires_token():
    r = client.get("/predict", params={"store": "store_1", "days": 3})
    assert r.status_code == 401


class FakeCursor:
    def __init__(self):
        self.one = None
        self.many = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        if "SELECT DISTINCT store" in sql:
            self.many = [{"store": "store_1"}, {"store": "store_2"}]
        elif "SELECT 1 FROM daily_sales" in sql:
            self.one = {"1": 1} if params[0] in {"store_1", "store_2"} else None
        elif "SELECT SUM(surplus)" in sql:
            self.one = {
                "surplus_units": 12.5,
                "sales_units": 100.0,
                "surplus_days": 5,
                "total_days": 10,
                "last_date": date(2023, 11, 30),
            }
        elif "SELECT sale_date, sales, surplus" in sql:
            self.many = [
                {"sale_date": date(2023, 11, 30), "sales": 10.0, "surplus": 2.0},
                {"sale_date": date(2023, 11, 29), "sales": 9.0, "surplus": None},
            ]
        else:
            self.one = None
            self.many = []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


def fake_db(monkeypatch):
    cursor = FakeCursor()

    @contextmanager
    def _fake_db():
        yield cursor

    monkeypatch.setattr(main, "db_cursor", _fake_db)
    return cursor


def test_stores_returns_list(monkeypatch):
    fake_db(monkeypatch)
    response = client.get("/stores", headers=auth_headers())
    assert response.status_code == 200
    assert response.json()["stores"] == ["store_1", "store_2"]


def test_history_returns_chronological_rows(monkeypatch):
    fake_db(monkeypatch)
    response = client.get(
        "/history",
        params={"store": "store_1", "days": 2},
        headers=auth_headers(),
    )
    assert response.status_code == 200
    history = response.json()["history"]
    assert [row["date"] for row in history] == ["2023-11-29", "2023-11-30"]
    assert history[0]["surplus"] is None


def test_stats_returns_aggregates(monkeypatch):
    fake_db(monkeypatch)
    response = client.get(
        "/stats",
        params={"store": "store_1", "days": 10},
        headers=auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["avg_daily_surplus"] == 2.5
    assert body["avg_daily_sales"] == 10.0
    assert body["last_date"] == "2023-11-30"


def test_stats_unknown_store_returns_404(monkeypatch):
    fake_db(monkeypatch)
    response = client.get(
        "/stats",
        params={"store": "unknown", "days": 10},
        headers=auth_headers(),
    )
    assert response.status_code == 404
