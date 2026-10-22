FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential нужен cmdstanpy, чтобы скомпилировать Stan-модель Prophet.
# Если Prophet установить не удастся, сервис продолжит работать на fallback-модели.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-prophet.txt ./
RUN pip install -r requirements.txt \
    && (pip install -r requirements-prophet.txt \
        || echo "Prophet is unavailable; the fallback forecaster will be used")

# Компилируем Stan-модель заранее, чтобы первый /predict не ждал сборку.
RUN python -c "import pandas as pd; from prophet import Prophet; m = Prophet(); m.fit(pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=20), 'y': list(range(20))}))" \
    || echo "Stan model was not precompiled"

COPY app ./app

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
