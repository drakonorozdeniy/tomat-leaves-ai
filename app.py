
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.disease_model import ModelNotReadyError, PyTorchDiseaseModel
from src.expert_system import ProductionExpertSystem
from src.external_detector import detect_plant_or_leaf
from src.preprocessing import open_rgb_image
from src.utils import disease_to_ru, format_percent


REGIONS = ["Симферополь", "Ялта", "Севастополь", "Бахчисарай", "Евпатория", "Керчь"]
HUMIDITY_LEVELS = ["низкая", "средняя", "высокая"]
SEASONS = ["весна", "лето", "осень", "зима"]
RULE_EXPLANATIONS = {
    "R1": "на изображении не обнаружен лист растения",
    "R2": "уверенность модели слишком низкая, явные признаки болезни не обнаружены",
    "R3": "результат неуверенный, нужно фото лучшего качества",
    "R4": "есть возможные признаки заболевания, нужна повторная проверка",
    "R5": "модель уверенно определила заболевание",
    "R6": "при фитофторозе и высокой влажности риск повышается до высокого",
    "R7": "для альтернариоза летом установлен средний уровень риска",
    "R8": "модель уверенно определила здоровый лист",
}


@st.cache_resource(show_spinner=False)
def get_expert_system() -> ProductionExpertSystem:
    return ProductionExpertSystem()


def render_text_metric(container, label: str, value: str) -> None:
    container.markdown(f"**{label}**")
    container.markdown(f"### {value}")


def render_external_result(result: dict) -> None:
    st.subheader("1. Внешняя модель ИИ: проверка наличия листа")
    columns = st.columns(4)
    columns[0].metric(
        "Лист найден",
        "Да" if result["object_detected"] else "Нет",
    )
    columns[1].metric("Уверенность", format_percent(result["object_confidence"]))
    columns[2].metric("YOLOv8", format_percent(result.get("yolo_confidence", 0.0)))
    columns[3].metric("Метод", result.get("detector", "YOLOv8"))


def render_disease_result(result: dict) -> None:
    st.subheader("2. Собственная PyTorch-модель")
    columns = st.columns(2)
    render_text_metric(columns[0], "Заболевание", result["disease_ru"])
    render_text_metric(columns[1], "Уверенность", format_percent(result["confidence"]))

    probability_rows = [
        {
            "Класс": disease_to_ru(label),
            "Метка": label,
            "Вероятность": probability,
        }
        for label, probability in result["probabilities"].items()
    ]
    probability_frame = pd.DataFrame(probability_rows)
    st.dataframe(probability_frame, use_container_width=True, hide_index=True)


def render_expert_result(result: dict) -> None:
    st.subheader("3. Экспертная система")
    columns = st.columns(3)
    render_text_metric(columns[0], "Итог", result["status_ru"])
    render_text_metric(columns[1], "Диагноз", result["diagnosis"])
    render_text_metric(columns[2], "Риск", result["risk_level"])

    if result["final_status"] == "diagnosed":
        st.success(result["explanation"])
    elif result["final_status"] == "no_object":
        st.error(result["explanation"])
    else:
        st.warning(result["explanation"])

    if result["rules_fired"]:
        st.markdown("**Сработавшие правила:**")
        for rule_id in result["rules_fired"]:
            explanation = RULE_EXPLANATIONS.get(rule_id, "описание правила не найдено")
            st.write(f"- {rule_id}: {explanation}.")
    else:
        st.write("Сработавшие правила: нет")

    if result["recommendations"]:
        st.markdown("**Рекомендации пользователю:**")
        for recommendation in result["recommendations"]:
            st.write(f"- {recommendation}")


def build_human_summary(
    external_result: dict,
    disease_result: dict | None,
    expert_result: dict,
) -> str:
    if not external_result["object_detected"]:
        return (
            "На фотографии не удалось надежно обнаружить лист растения. "
            "Экспертная система рекомендует загрузить другое фото."
        )

    if disease_result is None:
        return (
            "На фотографии обнаружен объект, похожий на лист растения. "
            "Для полной диагностики нужно обучить собственную PyTorch-модель."
        )

    diagnosis = disease_result["disease_ru"].lower()
    confidence = format_percent(disease_result["confidence"])
    risk = expert_result["risk_level"]
    rules = ", ".join(expert_result["rules_fired"]) or "нет"
    recommendations = " ".join(expert_result["recommendations"][:3])

    return (
        "На фотографии обнаружен лист растения. "
        f"Собственная модель определила класс: {diagnosis} с уверенностью {confidence}. "
        f"Экспертная система установила уровень риска: {risk}. "
        f"Сработавшие правила: {rules}. "
        f"{recommendations}"
    )


def main() -> None:
    st.set_page_config(
        page_title="Диагностика заболеваний томатов",
        layout="wide",
    )

    st.title("Диагностика заболеваний томатов")

    with st.sidebar:
        st.header("Условия выращивания")
        region = st.selectbox("Регион Крыма", REGIONS)
        humidity = st.selectbox("Влажность", HUMIDITY_LEVELS, index=1)
        season = st.selectbox("Сезон", SEASONS, index=1)

    uploaded_file = st.file_uploader(
        "Загрузите изображение листа томата",
        type=["jpg", "jpeg", "png", "webp"],
    )
    run_button = st.button("Провести диагностику", type="primary")

    if not run_button:
        return

    if uploaded_file is None:
        st.warning("Сначала загрузите изображение листа томата.")
        return

    try:
        image = open_rgb_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    left_column, right_column = st.columns([1, 1.2])
    with left_column:
        st.image(image, caption="Загруженное изображение", width="stretch")

    with right_column:
        external_result = detect_plant_or_leaf(image)
        render_external_result(external_result)

        disease_result: dict | None = None
        expert_result: dict | None = None

        if not external_result["object_detected"]:
            facts = {
                "object_detected": False,
                "object_confidence": external_result["object_confidence"],
                "disease": "unknown",
                "disease_confidence": 0.0,
                "region": region,
                "humidity": humidity,
                "season": season,
            }
            expert_result = get_expert_system().infer(facts)
            render_expert_result(expert_result)
        else:
            try:
                disease_model = PyTorchDiseaseModel()
                disease_result = disease_model.predict(image)
                render_disease_result(disease_result)
            except ModelNotReadyError as exc:
                st.warning(str(exc))

            if disease_result is not None:
                facts = {
                    "object_detected": external_result["object_detected"],
                    "object_confidence": external_result["object_confidence"],
                    "disease": disease_result["disease"],
                    "disease_confidence": disease_result["confidence"],
                    "region": region,
                    "humidity": humidity,
                    "season": season,
                }
                expert_result = get_expert_system().infer(facts)
                render_expert_result(expert_result)

        if expert_result is not None:
            st.subheader("Итоговое пояснение")
            st.write(build_human_summary(external_result, disease_result, expert_result))


if __name__ == "__main__":
    main()
