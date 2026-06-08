
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from src.preprocessing import image_is_valid, open_rgb_image
from src.utils import clamp


PLANT_LIKE_CLASSES = {
    "potted plant",
    "plant",
    "leaf",
    "flower",
    "tree",
    "broccoli",
}


@lru_cache(maxsize=1)
def _load_yolo(model_name: str) -> Any | None:
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except Exception:
        return None


def _quality_fallback(image: Image.Image) -> dict[str, Any]:
    is_valid, message = image_is_valid(image)
    if not is_valid:
        return {
            "object_detected": False,
            "object_confidence": 0.0,
            "message": message,
        }

    array = np.asarray(image).astype("float32")
    brightness = float(array.mean())
    color_std = float(array.std())
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    green_mask = (green > red * 1.05) & (green > blue * 1.05) & (green > 45)
    green_ratio = float(green_mask.mean())

    if brightness < 20:
        return {
            "object_detected": False,
            "object_confidence": 0.05,
            "message": "Изображение слишком темное для надежной диагностики.",
        }

    if color_std < 5:
        return {
            "object_detected": False,
            "object_confidence": 0.05,
            "message": "Изображение почти одноцветное, признаки листа не выделяются.",
        }

    confidence = clamp(0.35 + green_ratio * 0.55 + min(color_std / 255.0, 0.15))
    if green_ratio >= 0.03:
        message = (
            "YOLOv8 не дал надежного совпадения, но резервная проверка нашла "
            "достаточное качество изображения и зеленые области, похожие на лист."
        )
    else:
        message = (
            "YOLOv8 не дал надежного совпадения. Резервная проверка качества "
            "пройдена, но объект требует визуального контроля."
        )

    return {
        "object_detected": True,
        "object_confidence": confidence,
        "message": message,
    }


def detect_plant_or_leaf(
    image: Image.Image,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    use_yolo: bool = True,
) -> dict[str, Any]:
    """Проверить, есть ли на фото растение или лист."""
    rgb_image = open_rgb_image(image)

    yolo_status = "YOLOv8 не запускалась."
    yolo_confidence = 0.0
    detections: list[dict[str, Any]] = []

    if use_yolo:
        yolo_model = _load_yolo(model_name)
        if yolo_model is not None:
            try:
                results = yolo_model.predict(
                    source=np.asarray(rgb_image),
                    conf=confidence_threshold,
                    verbose=False,
                )
                for result in results:
                    names = result.names or {}
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        class_name = str(names.get(class_id, class_id)).lower()
                        detections.append(
                            {
                                "class_name": class_name,
                                "confidence": confidence,
                            }
                        )

                if detections:
                    best_any = max(detections, key=lambda item: item["confidence"])
                    yolo_confidence = clamp(best_any["confidence"])
                    yolo_status = (
                        "YOLOv8 была запущена, но среди найденных объектов нет "
                        f"подходящего класса растения/листа. Лучшее совпадение: "
                        f"{best_any['class_name']} ({yolo_confidence:.2f})."
                    )
                else:
                    yolo_status = (
                        "YOLOv8 была запущена, но не нашла объектов с заданным "
                        "порогом уверенности."
                    )

                plant_detections = [
                    item
                    for item in detections
                    if item["class_name"] in PLANT_LIKE_CLASSES
                ]
                if plant_detections:
                    best = max(plant_detections, key=lambda item: item["confidence"])
                    return {
                        "object_detected": True,
                        "object_confidence": clamp(best["confidence"]),
                        "message": (
                            "Внешняя модель YOLOv8 обнаружила объект, похожий на растение "
                            f"или лист: {best['class_name']}."
                        ),
                        "detector": "YOLOv8",
                        "yolo_status": "YOLOv8 нашла подходящий объект.",
                        "yolo_confidence": clamp(best["confidence"]),
                        "detections": detections,
                    }
            except Exception as exc:
                yolo_status = (
                    "YOLOv8 была вызвана, но не смогла завершить проверку: "
                    f"{exc}"
                )
        else:
            yolo_status = (
                "YOLOv8 недоступна или веса модели не загрузились. "
                "Использована резервная проверка изображения."
            )

    fallback_result = _quality_fallback(rgb_image)
    fallback_result["detector"] = "YOLOv8 + резервная проверка"
    fallback_result["yolo_status"] = yolo_status
    fallback_result["yolo_confidence"] = yolo_confidence
    fallback_result["detections"] = detections
    return fallback_result
