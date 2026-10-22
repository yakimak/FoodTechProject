import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import jwt
import pymysql
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

logger = logging.getLogger("surplus_guard.api")

app = FastAPI(title="Surplus Guard API", version="1.1.0")

ML_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001").rstrip("/")
ML_TIMEOUT = float(os.getenv("ML_TIMEOUT", "60"))
DB_HOST = os.getenv("DB_HOST", "mysql")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "surplus_guard")
DB_USER = os.getenv("DB_USER", "app")
DB_PASS = os.getenv("DB_PASS", "app")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me-0123456789abcdef")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "surplus-guard"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "3600"))

if JWT_SECRET == "dev-secret-change-me-0123456789abcdef":
    logger.warning(
        "JWT_SECRET is set to the development default. "
        "Set a strong secret before deploying to production."
    )

bearer_scheme = HTTPBearer(auto_error=False)


def get_conn():
    try:
        return pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
            read_timeout=30,
            write_timeout=30,
        )
    except pymysql.MySQLError as exc:
        logger.exception("Cannot connect to MySQL")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc


@contextmanager
def db_cursor():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            yield cur
    except pymysql.MySQLError as exc:
        logger.exception("Database query failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database query failed",
        ) from exc
    finally:
        conn.close()


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def ml_get(path: str, **params):
    try:
        r = requests.get(f"{ML_URL}{path}", params=params, timeout=ML_TIMEOUT)
    except requests.RequestException as exc:
        logger.exception("ML service request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ML service unavailable",
        ) from exc

    if r.status_code != 200:
        try:
            payload = r.json()
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        except ValueError:
            detail = r.text or r.reason
        raise HTTPException(status_code=r.status_code, detail=detail)

    try:
        return r.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ML service returned a non-JSON response",
        ) from exc


def _round(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def _ensure_store(cur, store: str) -> None:
    cur.execute("SELECT 1 FROM daily_sales WHERE store = %s LIMIT 1", (store,))
    if cur.fetchone() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown store: {store}",
        )


class LoginRequest(BaseModel):
    store_id: int = Field(default=1, ge=0)


@app.post("/login")
def login(req: LoginRequest):
    now = datetime.now(timezone.utc)
    payload = {
        "iss": JWT_ISSUER,
        "sub": str(req.store_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=JWT_TTL_SECONDS)).timestamp()),
        "store_id": req.store_id,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": JWT_TTL_SECONDS,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "api"}


@app.get("/stores")
def stores(_: dict = Depends(require_auth)):
    with db_cursor() as cur:
        cur.execute("SELECT DISTINCT store FROM daily_sales ORDER BY store")
        rows = cur.fetchall()
    return {"stores": [r["store"] for r in rows]}


@app.get("/predict")
def predict(
    store: str = Query(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    days: int = Query(7, ge=1, le=7),
    _: dict = Depends(require_auth),
):
    return ml_get("/predict", store=store, days=days)


@app.get("/history")
def history(
    store: str = Query(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    days: int = Query(90, ge=1, le=400),
    _: dict = Depends(require_auth),
):
    with db_cursor() as cur:
        _ensure_store(cur, store)
        cur.execute(
            """
            SELECT sale_date, sales, surplus
            FROM daily_sales
            WHERE store = %s
            ORDER BY sale_date DESC
            LIMIT %s
            """,
            (store, days),
        )
        rows = cur.fetchall()

    rows.reverse()
    return {
        "store": store,
        "history": [
            {
                "date": str(r["sale_date"])[:10],
                "sales": _round(r["sales"]),
                "surplus": _round(r["surplus"]),
            }
            for r in rows
        ],
    }


@app.get("/stats")
def stats(
    store: str = Query(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    days: int = Query(30, ge=1, le=400),
    _: dict = Depends(require_auth),
):
    with db_cursor() as cur:
        _ensure_store(cur, store)
        cur.execute(
            """
            SELECT SUM(surplus)  AS surplus_units,
                   SUM(sales)    AS sales_units,
                   COUNT(surplus) AS surplus_days,
                   COUNT(*)      AS total_days,
                   MAX(sale_date) AS last_date
            FROM daily_sales
            WHERE store = %s
              AND sale_date > DATE_SUB(
                    (SELECT MAX(sale_date) FROM daily_sales WHERE store = %s),
                    INTERVAL %s DAY
                  )
            """,
            (store, store, days),
        )
        row = cur.fetchone()

    surplus_units = float(row["surplus_units"] or 0)
    sales_units = float(row["sales_units"] or 0)
    surplus_days = int(row["surplus_days"] or 0)
    total_days = int(row["total_days"] or 0)

    return {
        "store": store,
        "period_days": days,
        "last_date": str(row["last_date"])[:10] if row["last_date"] else None,
        "surplus_units": round(surplus_units, 1),
        "avg_daily_surplus": round(surplus_units / surplus_days, 1) if surplus_days else 0.0,
        "sales_units": round(sales_units, 1),
        "avg_daily_sales": round(sales_units / total_days, 1) if total_days else 0.0,
        "surplus_days": surplus_days,
        "total_days": total_days,
    }


@app.get("/surplus-risk")
def surplus_risk(
    store: str = Query(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    days: int = Query(7, ge=1, le=7),
    _: dict = Depends(require_auth),
):
    data = ml_get("/predict", store=store, days=days)
    predictions = data.get("predictions") or []
    top = sorted(
        predictions,
        key=lambda x: float(x.get("predicted_surplus") or 0),
        reverse=True,
    )[:3]
    return {"store": store, "top_risk_days": top}
