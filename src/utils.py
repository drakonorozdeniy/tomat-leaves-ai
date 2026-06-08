
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_LABELS = ["early_blight", "healthy", "late_blight"]

DISEASE_RU = {
    "early_blight": "Альтернариоз",
    "healthy": "Здоровый лист",
    "late_blight": "Фитофтороз",
    "unknown": "Не определено",
}

STATUS_RU = {
    "diagnosed": "Заболевание диагностировано",
    "uncertain": "Требуется дополнительная проверка",
    "no_object": "Лист растения не обнаружен",
    "no_disease": "Признаки заболевания не обнаружены",
}

MODEL_MISSING_MESSAGE = (
    "Модель заболевания не найдена. Сначала обучите модель командой "
    "python training/train.py --data dataset"
)


def load_json(path: str | Path, default: Any | None = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: Any) -> None:
    """Сохранить данные в JSON с читаемым форматированием."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def disease_to_ru(disease: str | None) -> str:
    """Вернуть русское название класса заболевания."""
    if not disease:
        return DISEASE_RU["unknown"]
    return DISEASE_RU.get(str(disease), str(disease))


def format_percent(value: float | int | None) -> str:
    """Представить вероятность в процентах."""
    if value is None:
        return "0%"
    return f"{float(value) * 100:.1f}%"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Ограничить число заданным диапазоном."""
    return max(low, min(high, float(value)))
