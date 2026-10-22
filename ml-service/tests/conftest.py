import os
from pathlib import Path

# Тесты не должны зависеть от обученной модели в репозитории.
os.environ.setdefault(
    "MODEL_PATH",
    str(Path(__file__).with_name("missing-model.pkl")),
)
