"""Обучение моделей прогноза и сохранение bundle в models/forecast.pkl."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

# Позволяет запускать и `python ml-service/app/train.py`, и `python -m app.train`.
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from app.naive import NaiveForecaster  # noqa: E402

logger = logging.getLogger("surplus_guard.train")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

try:
    from prophet import Prophet
except Exception as exc:  # Prophet может падать и на этапе импорта
    Prophet = None
    PROPHET_IMPORT_ERROR: Exception | None = exc
else:
    PROPHET_IMPORT_ERROR = None


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://app:app@localhost:3306/surplus_guard?charset=utf8mb4",
    )


def _default_model_path() -> str:
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        return explicit
    if os.path.isdir("/app/models"):
        return "/app/models/forecast.pkl"
    return str(ML_ROOT.parent / "models" / "forecast.pkl")


DB = _database_url()
MODEL_PATH = _default_model_path()


def _make_prophet(yearly: bool) -> "Prophet":
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        daily_seasonality=False,
    )
    try:
        model.add_country_holidays(country_name="DE")
    except Exception:  # без пакета holidays модель всё ещё полезна
        logger.warning("Cannot add country holidays, continuing without them")
    return model


def _fit(df: pd.DataFrame, yearly: bool, label: str) -> tuple[object, str]:
    """Обучает Prophet, а при любой проблеме откатывается на NaiveForecaster."""
    if Prophet is not None and len(df) >= 2:
        try:
            return _make_prophet(yearly=yearly).fit(df), "prophet"
        except Exception:  # гарантируем работоспособность сервиса
            logger.exception("Prophet fit failed for %s, using fallback model", label)
    elif Prophet is None:
        logger.warning(
            "Prophet is unavailable (%s), using fallback model for %s",
            PROPHET_IMPORT_ERROR,
            label,
        )
    return NaiveForecaster.fit(df), "naive"


def _prepare(store_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    data = store_frame[["sale_date", column]].copy()
    data = data.dropna(subset=[column])
    data = data.rename(columns={"sale_date": "ds", column: "y"})
    data["ds"] = pd.to_datetime(data["ds"])
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["y"])
    return (
        data.sort_values("ds")
        .drop_duplicates(subset=["ds"], keep="last")
        .reset_index(drop=True)
    )


def _serialize(model: object) -> dict:
    """Сериализует Prophet в JSON, чтобы bundle не зависел от путей cmdstanpy."""
    if hasattr(model, "to_json"):
        try:
            return {"kind": "prophet", "json": model.to_json()}
        except Exception:  # старые версии Prophet могут не уметь to_json
            logger.exception("Cannot serialize Prophet model to JSON")
    return {"kind": "object", "model": model}


def train() -> None:
    engine = create_engine(DB, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT sale_date, store, sales, surplus
                    FROM daily_sales
                    ORDER BY sale_date, store
                    """
                ),
                conn,
                parse_dates=["sale_date"],
            )
    finally:
        engine.dispose()

    if df.empty:
        raise RuntimeError("daily_sales is empty, run etl/load.py first")

    required = {"sale_date", "store", "sales", "surplus"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"daily_sales is missing columns: {sorted(missing)}")

    sales_models: dict[str, object] = {}
    surplus_models: dict[str, object] = {}
    meta: dict[str, dict] = {}

    for store, g in df.groupby("store", sort=True):
        sales_df = _prepare(g, "sales")
        if sales_df.empty:
            logger.warning("%s: no sales data, skipping", store)
            continue

        history_days = (sales_df["ds"].max() - sales_df["ds"].min()).days
        sales_model, sales_engine = _fit(
            sales_df,
            yearly=history_days > 365,
            label=f"{store}/sales",
        )
        sales_models[store] = _serialize(sales_model)

        surplus_df = _prepare(g, "surplus")
        surplus_engine = "none"
        if surplus_df.empty:
            logger.warning(
                "%s: no surplus history, surplus forecasts will be zero", store
            )
        else:
            surplus_model, surplus_engine = _fit(
                surplus_df,
                yearly=False,
                label=f"{store}/surplus",
            )
            surplus_models[store] = _serialize(surplus_model)

        meta[store] = {
            "last_date": sales_df["ds"].max().strftime("%Y-%m-%d"),
            "sales_days": len(sales_df),
            "surplus_days": len(surplus_df),
            "sales_engine": sales_engine,
            "surplus_engine": surplus_engine,
        }
        logger.info(
            "%s: sales days=%d (%s), surplus days=%d (%s)",
            store,
            len(sales_df),
            sales_engine,
            len(surplus_df),
            surplus_engine,
        )

    if not sales_models:
        raise RuntimeError("no models were trained")

    bundle = {"sales": sales_models, "surplus": surplus_models, "meta": meta}

    path = Path(MODEL_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(bundle, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info("Saved models for %d stores to %s", len(sales_models), path)


if __name__ == "__main__":
    train()
