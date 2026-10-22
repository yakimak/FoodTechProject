"""Простой fallback-прогноз на случай, если Prophet недоступен.

Класс повторяет минимальный интерфейс Prophet, который использует сервис:
``make_future_dataframe`` и ``predict``. Этого достаточно, чтобы ML-сервис
продолжал работать без тяжёлых зависимостей.

Модель хранит только простые Python-типы (float/dict/list/str), поэтому
pickle не зависит от версии pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveForecaster:
    """Среднее значение + поправка на день недели."""

    def __init__(
        self,
        mean: float,
        weekday_effects: dict[int, float],
        last_date: str,
        history_dates: list[str],
    ) -> None:
        self.mean = float(mean)
        self.weekday_effects = {int(k): float(v) for k, v in weekday_effects.items()}
        self.last_date = str(last_date)
        self.history_dates = [str(date) for date in history_dates]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "NaiveForecaster":
        data = df[["ds", "y"]].dropna().copy()
        data["ds"] = pd.to_datetime(data["ds"])
        data = (
            data.sort_values("ds")
            .drop_duplicates(subset=["ds"], keep="last")
            .reset_index(drop=True)
        )
        if data.empty:
            raise ValueError("cannot fit NaiveForecaster on an empty dataset")

        mean = float(data["y"].mean())
        weekday_effects: dict[int, float] = {day: 0.0 for day in range(7)}
        if len(data) >= 14:
            weekday_means = data.groupby(data["ds"].dt.day_of_week)["y"].mean()
            weekday_effects = {
                day: float(weekday_means.get(day, mean) - mean) for day in range(7)
            }

        return cls(
            mean=mean,
            weekday_effects=weekday_effects,
            last_date=data["ds"].max().strftime("%Y-%m-%d"),
            history_dates=data["ds"].dt.strftime("%Y-%m-%d").tolist(),
        )

    def make_future_dataframe(
        self,
        periods: int,
        freq: str = "D",
        include_history: bool = False,
    ) -> pd.DataFrame:
        if periods < 0:
            raise ValueError("periods must be non-negative")
        last_date_value = getattr(self, "last_date", None)
        if last_date_value:
            last_date = pd.Timestamp(last_date_value)
        else:
            # Совместимость с bundle, сохранёнными до появления last_date.
            history = getattr(self, "history", None)
            if history is None or history.empty:
                raise ValueError("NaiveForecaster has no history dates")
            last_date = pd.Timestamp(history["ds"].max())
        future = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq=freq,
        )
        frame = pd.DataFrame({"ds": future})
        if include_history:
            history = pd.DataFrame({"ds": pd.to_datetime(self.history_dates)})
            frame = pd.concat([history, frame], ignore_index=True)
        return frame

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        ds = pd.to_datetime(pd.Series(df["ds"])).reset_index(drop=True)
        yhat = np.array(
            [
                self.mean + self.weekday_effects.get(int(ts.day_of_week), 0.0)
                for ts in ds
            ],
            dtype=float,
        )
        return pd.DataFrame({"ds": ds, "yhat": yhat})
