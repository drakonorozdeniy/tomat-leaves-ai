
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.recommendations import get_recommendations
from src.utils import DATA_DIR, STATUS_RU, disease_to_ru, format_percent, load_json


class ProductionExpertSystem:
    """Интерпретатор правил из data/rules.json."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        recommendations_path: str | Path | None = None,
    ) -> None:
        self.rules_path = Path(rules_path) if rules_path else DATA_DIR / "rules.json"
        self.recommendations_path = (
            Path(recommendations_path)
            if recommendations_path
            else DATA_DIR / "recommendations.json"
        )
        self.rules = sorted(
            load_json(self.rules_path, default=[]) or [],
            key=lambda rule: int(rule.get("priority", 100)),
        )

    def infer(self, facts: dict[str, Any]) -> dict[str, Any]:
        """Применить правила к фактам и вернуть итоговое решение."""
        disease = facts.get("disease") or "unknown"
        result: dict[str, Any] = {
            "final_status": "uncertain",
            "risk_level": "низкий",
            "diagnosis": disease_to_ru(disease),
            "explanation": "",
            "rules_fired": [],
            "recommendations": [],
        }

        explanations: list[str] = []
        for rule in self.rules:
            if self._conditions_are_met(rule.get("conditions", []), facts):
                result["rules_fired"].append(rule.get("id", "unknown"))
                self._apply_actions(result, rule.get("actions", {}), facts)
                message = self._format_message(rule.get("actions", {}).get("message"), facts)
                if message:
                    explanations.append(f"{rule.get('id')}: {message}")
                if rule.get("actions", {}).get("stop"):
                    break

        if not result["rules_fired"]:
            explanations.append(
                "Ни одно специальное правило не сработало, поэтому результат требует повторной проверки."
            )

        result["explanation"] = " ".join(explanations)
        result["recommendations"] = self._select_recommendations(result, disease)
        result["status_ru"] = STATUS_RU.get(result["final_status"], result["final_status"])
        return result

    def _conditions_are_met(
        self, conditions: list[dict[str, Any]], facts: dict[str, Any]
    ) -> bool:
        return all(self._condition_is_met(condition, facts) for condition in conditions)

    def _condition_is_met(self, condition: dict[str, Any], facts: dict[str, Any]) -> bool:
        field = condition.get("field")
        operator = condition.get("operator")
        expected = condition.get("value")
        current = facts.get(field)

        if operator in {"lt", "lte", "gt", "gte"}:
            try:
                current_number = float(current)
                expected_number = float(expected)
            except (TypeError, ValueError):
                return False

            if operator == "lt":
                return current_number < expected_number
            if operator == "lte":
                return current_number <= expected_number
            if operator == "gt":
                return current_number > expected_number
            if operator == "gte":
                return current_number >= expected_number

        if operator == "eq":
            return self._normalize(current) == self._normalize(expected)
        if operator == "ne":
            return self._normalize(current) != self._normalize(expected)
        if operator == "in":
            return current in expected

        raise ValueError(f"Неизвестный оператор правила: {operator}")

    def _apply_actions(
        self,
        result: dict[str, Any],
        actions: dict[str, Any],
        facts: dict[str, Any],
    ) -> None:
        for field in ("final_status", "risk_level", "diagnosis"):
            if field in actions:
                result[field] = actions[field]

        if actions.get("diagnosis_from_disease"):
            result["diagnosis"] = disease_to_ru(facts.get("disease"))

    def _select_recommendations(self, result: dict[str, Any], disease: str) -> list[str]:
        if result["final_status"] == "no_object":
            return []

        if result["final_status"] == "no_disease":
            recommendation_key = "healthy"
        else:
            recommendation_key = disease

        return get_recommendations(
            recommendation_key,
            result["risk_level"],
            self.recommendations_path,
        )

    def _format_message(self, template: str | None, facts: dict[str, Any]) -> str:
        if not template:
            return ""

        context = dict(facts)
        context["disease_ru"] = disease_to_ru(facts.get("disease"))
        context["confidence_percent"] = format_percent(facts.get("disease_confidence"))
        try:
            return template.format(**context)
        except KeyError:
            return template

    @staticmethod
    def _normalize(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


def run_expert_system(facts: dict[str, Any]) -> dict[str, Any]:

    return ProductionExpertSystem().infer(facts)
