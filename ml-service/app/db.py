import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine


def build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = quote_plus(os.getenv("DB_USER", "app"))
    password = quote_plus(os.getenv("DB_PASS", "app"))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "surplus_guard")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


engine = create_engine(build_database_url(), pool_pre_ping=True)


def get_engine():
    return engine
