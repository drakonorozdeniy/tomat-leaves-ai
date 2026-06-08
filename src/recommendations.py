
from __future__ import annotations

from pathlib import Path

from src.utils import DATA_DIR, load_json


def get_recommendations(
    disease: str | None,
    risk_level: str = "низкий",
    recommendations_path: str | Path | None = None,
) -> list[str]:
    """Вернуть список советов для пользователя."""
    path = Path(recommendations_path) if recommendations_path else DATA_DIR / "recommendations.json"
    data = load_json(path, default={}) or {}

    disease_key = disease if disease in data else "healthy"
    if not disease_key or disease_key not in data:
        return []

    record = data[disease_key]
    recommendations = list(record.get("base", []))
    recommendations.extend(record.get("risk", {}).get(risk_level, []))


    unique: list[str] = []
    for item in recommendations:
        if item and item not in unique:
            unique.append(item)
    return unique
