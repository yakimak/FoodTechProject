"""ETL: CSV с продажами -> агрегированная таблица daily_sales в MySQL."""

import logging
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text, types

logger = logging.getLogger("surplus_guard.etl")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = quote_plus(os.getenv("DB_USER", "app"))
    password = quote_plus(os.getenv("DB_PASS", "app"))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "surplus_guard")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


def _default_source() -> str:
    explicit = os.getenv("DATA_PATH")
    if explicit:
        return explicit
    if os.path.isfile("/app/data/train.csv"):
        return "/app/data/train.csv"
    return str(Path(__file__).resolve().parents[1] / "data" / "train.csv")


DB = _database_url()
SOURCE = _default_source()
REQUIRED_COLUMNS = {
    "date",
    "store",
    "is_state_holiday",
    "temperature_mean",
    "precipitation_sum",
    "sales",
    "unsold",
    "ordered",
}

DTYPES = {
    "store": types.VARCHAR(64),
    "sale_date": types.Date(),
    "sales": types.Float(),
    "surplus": types.Float(),
    "ordered": types.Float(),
    "temperature_mean": types.Float(),
    "precipitation_sum": types.Float(),
    "is_holiday": types.Integer(),
}


def load() -> None:
    raw = pd.read_csv(SOURCE, parse_dates=["date"])

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"{SOURCE} is missing columns: {sorted(missing)}")

    raw = raw.dropna(subset=["sales"])
    raw["is_holiday"] = (
        raw["is_state_holiday"].fillna("normal_day") != "normal_day"
    ).astype(int)

    # unsold = излишки. Пропуски сохраняем как NULL, а не как 0:
    # в 2021 году излишков нет вообще, и 0 исказил бы статистику.
    daily = (
        raw.groupby(["date", "store"], as_index=False)
        .agg(
            sales=("sales", "sum"),
            surplus=("unsold", lambda s: s.sum(min_count=1)),
            ordered=("ordered", lambda s: s.sum(min_count=1)),
            temperature_mean=("temperature_mean", "mean"),
            precipitation_sum=("precipitation_sum", "sum"),
            is_holiday=("is_holiday", "max"),
        )
        .rename(columns={"date": "sale_date"})
        .sort_values(["store", "sale_date"])
        .reset_index(drop=True)
    )

    # Источник стандартизирован (z-оценки), поэтому значения могут быть отрицательными.
    # Для бизнес-метрик сдвигаем каждый ряд в положительную область по магазину:
    # это сохраняет колебания и сезонность, но убирает бессмысленные минус-продажи.
    for col in ("sales", "surplus", "ordered"):
        daily[col] = daily.groupby("store")[col].transform(
            lambda s: s - s.min() + 0.1 if s.notna().any() else s
        )

    # Явно приводим к date, чтобы MySQL-драйвер не терял тип DATE.
    daily["sale_date"] = pd.to_datetime(daily["sale_date"]).dt.date

    engine = create_engine(DB, pool_pre_ping=True)
    try:
        daily.to_sql(
            "daily_sales",
            engine,
            if_exists="replace",
            index=False,
            dtype=DTYPES,
        )
        with engine.begin() as conn:
            conn.execute(
                text("CREATE INDEX idx_store_date ON daily_sales (store, sale_date)")
            )
    finally:
        engine.dispose()

    logger.info(
        "Loaded %d store-days for %d stores into daily_sales",
        len(daily),
        daily["store"].nunique(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load()
