
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models

from src.preprocessing import get_eval_transforms, open_rgb_image
from src.utils import (
    DEFAULT_LABELS,
    DISEASE_RU,
    MODEL_MISSING_MESSAGE,
    MODELS_DIR,
    disease_to_ru,
    load_json,
)


class ModelNotReadyError(RuntimeError):
    """Ошибка, когда модель еще не обучена или не может быть загружена."""


def _labels_from_idx_mapping(mapping: dict[str, Any]) -> list[str]:
    return [str(mapping[key]) for key in sorted(mapping, key=lambda item: int(item))]


def load_labels(labels_path: str | Path | None = None) -> list[str]:
    """Загрузить список классов из labels.json."""
    path = Path(labels_path) if labels_path else MODELS_DIR / "labels.json"
    data = load_json(path, default=None)

    if data is None:
        return list(DEFAULT_LABELS)

    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        if "idx_to_class" in data:
            return _labels_from_idx_mapping(data["idx_to_class"])
        if "class_to_idx" in data:
            return [
                class_name
                for class_name, _ in sorted(
                    data["class_to_idx"].items(), key=lambda item: int(item[1])
                )
            ]

        numeric_keys = {
            key: value for key, value in data.items() if str(key).isdigit()
        }
        if numeric_keys:
            return _labels_from_idx_mapping(numeric_keys)

    return list(DEFAULT_LABELS)


def _weights_for_architecture(architecture: str, pretrained: bool) -> Any | None:
    if not pretrained:
        return None

    if architecture == "mobilenet_v3_small":
        return models.MobileNet_V3_Small_Weights.DEFAULT
    if architecture == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT

    raise ValueError(f"Неизвестная архитектура: {architecture}")


def create_model(
    num_classes: int,
    architecture: str = "mobilenet_v3_small",
    pretrained: bool = False,
) -> nn.Module:
    """Создать модель transfer learning и заменить последний слой."""
    weights = _weights_for_architecture(architecture, pretrained)

    try:
        if architecture == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model

        if architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model
    except Exception as exc:
        if pretrained:
            warnings.warn(
                "Не удалось загрузить предобученные веса torchvision. "
                "Модель будет создана без них.",
                RuntimeWarning,
            )
            return create_model(num_classes, architecture, pretrained=False)
        raise exc

    raise ValueError(f"Неизвестная архитектура: {architecture}")


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class PyTorchDiseaseModel:
    """Класс для загрузки модели и предсказания заболевания по изображению."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        architecture: str = "mobilenet_v3_small",
        image_size: int = 224,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "disease_model.pt"
        self.labels_path = Path(labels_path) if labels_path else MODELS_DIR / "labels.json"
        self.architecture = architecture
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.labels = load_labels(self.labels_path)
        self.model: nn.Module | None = None

    def load_model(self) -> None:
        """Загрузить веса модели из models/disease_model.pt."""
        if not self.model_path.exists():
            raise ModelNotReadyError(MODEL_MISSING_MESSAGE)

        try:
            checkpoint = _torch_load(self.model_path, self.device)

            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
                self.architecture = checkpoint.get("architecture", self.architecture)
                self.image_size = int(checkpoint.get("image_size", self.image_size))
                if "idx_to_class" in checkpoint:
                    self.labels = _labels_from_idx_mapping(checkpoint["idx_to_class"])
                elif "labels" in checkpoint:
                    self.labels = [str(label) for label in checkpoint["labels"]]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model = create_model(
                num_classes=len(self.labels),
                architecture=self.architecture,
                pretrained=False,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
        except ModelNotReadyError:
            raise
        except Exception as exc:
            raise ModelNotReadyError(f"Не удалось загрузить модель заболевания: {exc}") from exc

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Подготовить изображение к подаче в нейросеть."""
        rgb_image = open_rgb_image(image)
        transform = get_eval_transforms(self.image_size)
        return transform(rgb_image).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Вернуть диагноз, уверенность и вероятности по всем классам."""
        if self.model is None:
            self.load_model()

        assert self.model is not None
        tensor = self.preprocess_image(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()

        probabilities = {
            label: float(probabilities_tensor[index])
            for index, label in enumerate(self.labels)
        }
        best_index = int(torch.argmax(probabilities_tensor).item())
        disease = self.labels[best_index]
        confidence = float(probabilities_tensor[best_index])

        return {
            "disease": disease,
            "disease_ru": disease_to_ru(disease),
            "confidence": confidence,
            "probabilities": probabilities,
            "disease_ru_map": DISEASE_RU,
        }
