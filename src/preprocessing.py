
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Аугментации для обучающей выборки."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Детерминированные преобразования для проверки и предсказания."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def open_rgb_image(image: Image.Image | str | Path | bytes | BinaryIO) -> Image.Image:
    """Открыть изображение из разных источников и привести к RGB."""
    try:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")

        if isinstance(image, bytes):
            return Image.open(BytesIO(image)).convert("RGB")

        return Image.open(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Не удалось открыть изображение. Проверьте формат файла.") from exc


def image_is_valid(image: Image.Image, min_size: int = 64) -> tuple[bool, str]:
    width, height = image.size
    if width < min_size or height < min_size:
        return False, f"Изображение слишком маленькое: {width}x{height}."
    return True, "Изображение подходит для анализа."
