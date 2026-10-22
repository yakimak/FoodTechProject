import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger("surplus_guard.ml")


def _default_model_path() -> str:
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        return explicit
    if os.path.isdir("/app/models"):
        return "/app/models/forecast.pkl"
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "models" / "forecast.pkl")


MODEL_PATH = _default_model_path()
BUNDLE: dict | None = None
_LAST_ATTEMPT_MTIME: float | None = None


def _deserialize(model):
    """Превращает сохранённое представление модели в объект с методом predict."""
    if isinstance(model, dict):
        kind = model.get("kind")
        if kind == "prophet":
            from prophet import Prophet

            return Prophet.from_json(model["json"])
        if kind in {"naive", "object"}:
            return model["model"]
        raise ValueError(f"unknown model payload: {kind!r}")

    # Обратная совместимость со старым forecast.pkl, где лежали сами объекты
    # Prophet. Перегоняем их через JSON, чтобы избавиться от абсолютных путей
    # cmdstanpy, оставшихся от контейнера, в котором обучалась модель.
    module = getattr(model.__class__, "__module__", "")
    if module.startswith("prophet"):
        try:
            from prophet import Prophet

            return Prophet.from_json(model.to_json())
        except Exception:  # пробуем использовать объект как есть
            logger.exception("Cannot normalize a pickled Prophet model via JSON")
    return model


def _load(force: bool = False) -> None:
    global BUNDLE, _LAST_ATTEMPT_MTIME

    try:
        mtime = os.path.getmtime(MODEL_PATH)
    except OSError:
        if BUNDLE is not None:
            logger.warning("Model file %s disappeared", MODEL_PATH)
        BUNDLE = None
        _LAST_ATTEMPT_MTIME = None
        return

    if not force and mtime == _LAST_ATTEMPT_MTIME:
        return
    _LAST_ATTEMPT_MTIME = mtime

    try:
        raw = joblib.load(MODEL_PATH)
        if not isinstance(raw, dict) or not raw.get("sales"):
            raise ValueError("invalid model bundle: 'sales' is missing")

        sales = {store: _deserialize(model) for store, model in raw["sales"].items()}
        surplus: dict[str, object] = {}
        for store, model in (raw.get("surplus") or {}).items():
            try:
                surplus[store] = _deserialize(model)
            except Exception:  # без излишков можно работать
                logger.exception("Cannot load surplus model for %s", store)

        BUNDLE = {
            "sales": sales,
            "surplus": surplus,
            "meta": raw.get("meta") or {},
        }
        logger.info("Model loaded from %s (%d stores)", MODEL_PATH, len(sales))
    except Exception:  # повреждённый файл не должен ронять сервис
        logger.exception("Model not loaded from %s", MODEL_PATH)
        BUNDLE = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(title="Surplus Guard ML Service", version="1.1.0", lifespan=lifespan)


def _require_bundle() -> dict:
    # Модель могла переобучиться уже после старта сервиса. _load сам проверяет
    # mtime и сразу выходит, если файл не менялся.
    _load()
    if BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Run init or check the service logs.",
        )
    return BUNDLE


def _positive_round(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return round(max(value, 0.0), 2)


def _last_training_date(bundle: dict, store: str, model) -> pd.Timestamp:
    meta = (bundle.get("meta") or {}).get(store) or {}
    last_date = meta.get("last_date")
    if last_date:
        try:
            return pd.Timestamp(last_date)
        except (TypeError, ValueError):
            logger.warning("Invalid last_date for %s: %r", store, last_date)

    model_last_date = getattr(model, "last_date", None)
    if model_last_date:
        try:
            return pd.Timestamp(model_last_date)
        except (TypeError, ValueError):
            logger.warning("Invalid model last_date for %s: %r", store, model_last_date)

    history = getattr(model, "history", None)
    if isinstance(history, pd.DataFrame) and not history.empty and "ds" in history:
        return pd.Timestamp(history["ds"].max())

    raise HTTPException(
        status_code=500,
        detail=f"Model metadata for {store} has no last training date",
    )


@app.get("/health")
def health():
    _load()
    return {
        "status": "ok",
        "model_loaded": BUNDLE is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/stores")
def stores():
    return {"stores": sorted(_require_bundle()["sales"].keys())}


@app.get("/predict")
def predict(
    store: str = Query(..., min_length=1, max_length=64),
    days: int = Query(7, ge=1, le=30),
):
    bundle = _require_bundle()
    sales_model = bundle["sales"].get(store)
    if sales_model is None:
        raise HTTPException(status_code=404, detail=f"unknown store: {store}")

    last_date = _last_training_date(bundle, store, sales_model)
    future = pd.DataFrame(
        {
            "ds": pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=days,
                freq="D",
            )
        }
    )

    sales_yhat = sales_model.predict(future)["yhat"].to_numpy(dtype=float)

    surplus_model = bundle.get("surplus", {}).get(store)
    surplus_yhat = np.zeros(days, dtype=float)
    if surplus_model is not None:
        try:
            surplus_yhat = surplus_model.predict(future)["yhat"].to_numpy(dtype=float)
        except Exception:  # прогноз продаж важнее, излишки можно обнулить
            logger.exception("Surplus forecast failed for %s, using zeros", store)

    return {
        "store": store,
        "last_training_date": last_date.strftime("%Y-%m-%d"),
        "predictions": [
            {
                "date": ds.strftime("%Y-%m-%d"),
                "predicted_sales": _positive_round(sales),
                "predicted_surplus": _positive_round(surplus),
            }
            for ds, sales, surplus in zip(future["ds"], sales_yhat, surplus_yhat)
        ],
    }
