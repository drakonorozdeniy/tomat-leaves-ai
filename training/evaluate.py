
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.disease_model import ModelNotReadyError, PyTorchDiseaseModel  # noqa: E402
from src.preprocessing import get_eval_transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка модели болезней томата.")
    parser.add_argument("--data", type=Path, required=True, help="Путь к папке dataset")
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Какую часть датасета оценивать",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "disease_model.pt",
        help="Путь к сохраненной модели",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "models" / "labels.json",
        help="Путь к labels.json",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча")
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "confusion_matrix.png",
        help="Куда сохранить confusion matrix",
    )
    return parser.parse_args()


def save_confusion_matrix(
    matrix: list[list[int]] | torch.Tensor,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Confusion matrix")

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            ax.text(
                col_index,
                row_index,
                str(matrix[row_index][col_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = args.data
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    split_dir = data_dir / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Папка выборки не найдена: {split_dir}")

    classifier = PyTorchDiseaseModel(args.model, args.labels)
    try:
        classifier.load_model()
    except ModelNotReadyError as exc:
        print(exc)
        return 1

    dataset = ImageFolder(
        split_dir,
        transform=get_eval_transforms(classifier.image_size),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []
    model_labels = classifier.labels
    model = classifier.model
    assert model is not None

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(classifier.device)
            logits = model(images)
            predicted_indexes = logits.argmax(dim=1).cpu().tolist()

            for predicted_index in predicted_indexes:
                predicted_label = model_labels[predicted_index]
                if predicted_label not in dataset.class_to_idx:
                    raise ValueError(
                        f"Класс модели {predicted_label} отсутствует в датасете."
                    )
                y_pred.append(dataset.class_to_idx[predicted_label])

            y_true.extend(targets.tolist())

    labels = list(range(len(dataset.classes)))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=dataset.classes,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=dataset.classes, columns=dataset.classes))

    save_confusion_matrix(matrix, dataset.classes, args.confusion_output)
    print(f"Confusion matrix сохранена: {args.confusion_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
