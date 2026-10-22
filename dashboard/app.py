import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Surplus Guard", page_icon="🍽", layout="wide")

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 25
TOKEN_TTL = 1800


@st.cache_data(ttl=TOKEN_TTL, show_spinner=False)
def get_token() -> str:
    try:
        response = requests.post(
            f"{API}/login",
            json={"store_id": 1},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"API недоступен: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(
            f"не удалось получить токен ({response.status_code}): {response.text[:200]}"
        )
    return response.json()["access_token"]


def api_get(path: str, **params):
    token = get_token()
    for attempt in range(2):
        try:
            response = requests.get(
                f"{API}{path}",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"API недоступен: {exc}") from exc

        if response.status_code == 401 and attempt == 0:
            # Токен мог истечь — сбрасываем кэш и пробуем ещё раз.
            get_token.clear()
            token = get_token()
            continue

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail", detail)
            except ValueError:
                pass
            raise RuntimeError(f"API вернул {response.status_code}: {detail}")

        return response.json()

    raise RuntimeError("не удалось авторизоваться в API")


def _empty_frame(columns):
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})
    frame.index = pd.DatetimeIndex([], name="date")
    return frame


def to_frame(rows, columns):
    df = pd.DataFrame(rows or [])
    if df.empty or "date" not in df.columns:
        return _empty_frame(columns)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.set_index("date")[columns]


st.title("🍽 Surplus Guard — прогноз продаж и излишков")
st.caption(
    "Реальные данные сети магазинов (Green-AI-Hub-Mittelstand, 02.08.2021 – 30.11.2023). "
    "Значения стандартизированы, поэтому единицы условные (не кг и не рубли)."
)

try:
    stores = [store for store in api_get("/stores")["stores"] if store]
except Exception as exc:
    st.error(f"API недоступен или авторизация не удалась: {exc}")
    st.stop()

if not stores:
    st.warning("В базе нет магазинов. Запустите `docker compose up init`.")
    st.stop()

with st.sidebar:
    st.header("Параметры")
    store = st.selectbox("Магазин", stores)
    horizon = st.slider("Горизонт прогноза, дней", 1, 7, 7)
    st.divider()
    st.caption("Прогноз строится на дни после последней даты в данных.")

try:
    stats = api_get("/stats", store=store, days=30)
    hist = api_get("/history", store=store, days=120)
    pred = api_get("/predict", store=store, days=horizon)
    risk = api_get("/surplus-risk", store=store, days=horizon)
except Exception as exc:
    st.error(f"Ошибка запроса к API: {exc}")
    st.stop()

st.caption(f"Последняя дата в данных: {stats.get('last_date') or '—'}")

# ---- KPI ----
pred_df = to_frame(
    pred.get("predictions", []),
    ["predicted_sales", "predicted_surplus"],
)
pred_sales = float(pred_df["predicted_sales"].sum())
pred_surplus = float(pred_df["predicted_surplus"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Излишки за 30 дней, усл. ед.", f"{float(stats.get('surplus_units', 0)):.1f}")
c2.metric("Излишки в среднем в день", f"{float(stats.get('avg_daily_surplus', 0)):.1f}")
c3.metric(f"Прогноз продаж, {horizon} дн.", f"{pred_sales:.1f}")
c4.metric(f"Прогноз излишков, {horizon} дн.", f"{pred_surplus:.1f}")

st.divider()

# ---- Продажи: факт + прогноз ----
hist_df = to_frame(hist.get("history", []), ["sales", "surplus"])
chart_index = hist_df.index.union(pred_df.index)

st.subheader("📈 Продажи: факт и прогноз")
if len(chart_index):
    sales_chart = pd.DataFrame(index=chart_index)
    sales_chart["Факт"] = hist_df["sales"]
    sales_chart["Прогноз"] = pred_df["predicted_sales"]
    st.line_chart(sales_chart, height=320)
else:
    st.info("Нет данных для построения графика.")

# ---- Излишки: факт + прогноз ----
st.subheader("🗑️ Излишки: факт и прогноз")
if len(chart_index):
    surplus_chart = pd.DataFrame(index=chart_index)
    surplus_chart["Факт"] = hist_df["surplus"]
    surplus_chart["Прогноз"] = pred_df["predicted_surplus"]
    st.line_chart(surplus_chart, height=260)
else:
    st.info("Нет данных для построения графика.")

# ---- Риски ----
st.subheader("⚠️ Дни с наибольшим прогнозируемым излишком")
risk_rows = risk.get("top_risk_days") or []
if risk_rows:
    risk_df = pd.DataFrame(risk_rows).rename(
        columns={
            "date": "Дата",
            "predicted_sales": "Прогноз продаж",
            "predicted_surplus": "Прогноз излишков",
        }
    )
    if "Дата" in risk_df.columns:
        risk_df["Дата"] = pd.to_datetime(
            risk_df["Дата"], errors="coerce"
        ).dt.strftime("%d.%m.%Y")
    columns = [
        column
        for column in ("Дата", "Прогноз продаж", "Прогноз излишков")
        if column in risk_df.columns
    ]
    st.dataframe(risk_df[columns], hide_index=True)
else:
    st.info("Прогноз не вернул данные о рисках излишков.")

with st.expander("ℹ️ Как строится прогноз"):
    st.markdown(
        """
        - **Модель продаж**: Prophet, отдельно для каждого магазина. Учитывает тренд,
          недельную и годовую сезонность и праздники Германии. Если Prophet
          недоступен, используется простая fallback-модель (среднее + день недели).
        - **Модель излишков**: Prophet, обучена только на днях, где известны излишки
          (в данных это 2022–2023 годы).
        - **Риск**: прогнозный объём излишков по дням; в таблице — дни с наибольшим значением.
        - **Единицы**: исходные значения стандартизированы и могут быть отрицательными.
          Поэтому каждый ряд сдвинут в положительную область по магазину
          (`значение − минимум + 0.1`); колебания и сезонность сохранены,
          но абсолютные значения условные — кг и рубли здесь не считаются.
        """
    )
