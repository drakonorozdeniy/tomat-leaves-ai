
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.disease_model import create_model  # noqa: E402
from src.preprocessing import get_eval_transforms, get_train_transforms  # noqa: E402
from src.utils import DEFAULT_LABELS, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение PyTorch-модели для диагностики болезней томата."
    )
    parser.add_argument("--data", type=Path, required=True, help="Путь к папке dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Количество эпох обучения")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-3, help="Скорость обучения")
    parser.add_argument(
        "--architecture",
        choices=["mobilenet_v3_small", "efficientnet_b0"],
        default="mobilenet_v3_small",
        help="Архитектура базовой модели",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Размер входа модели")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "disease_model.pt",
        help="Куда сохранить веса модели",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "labels.json",
        help="Куда сохранить соответствие индексов и классов",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Использовать предобученные веса torchvision",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Количество процессов DataLoader. Для Windows безопасно оставить 0.",
    )
    return parser.parse_args()


def validate_dataset(data_dir: Path) -> None:
    required_splits = ["train", "val"]
    missing = [split for split in required_splits if not (data_dir / split).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"В датасете отсутствуют папки: {joined}")


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, ImageFolder]:
    train_dataset = ImageFolder(
        data_dir / "train",
        transform=get_train_transforms(image_size),
    )
    val_dataset = ImageFolder(
        data_dir / "val",
        transform=get_eval_transforms(image_size),
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Классы в train и val должны совпадать.")

    if train_dataset.classes != DEFAULT_LABELS:
        print(
            "Предупреждение: порядок классов отличается от ожидаемого "
            f"{DEFAULT_LABELS}. Текущий порядок: {train_dataset.classes}"
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += batch_size

    average_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    train_dataset: ImageFolder,
    output_path: Path,
    labels_path: Path,
    architecture: str,
    image_size: int,
    val_accuracy: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    idx_to_class = {
        str(index): class_name
        for class_name, index in train_dataset.class_to_idx.items()
    }
    labels_payload = {
        "idx_to_class": idx_to_class,
        "class_to_idx": train_dataset.class_to_idx,
    }

    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "image_size": image_size,
            "idx_to_class": idx_to_class,
            "class_to_idx": train_dataset.class_to_idx,
            "labels": train_dataset.classes,
            "val_accuracy": val_accuracy,
        },
        output_path,
    )
    save_json(labels_path, labels_payload)


def main() -> int:
    args = parse_args()
    data_dir = args.data
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    validate_dataset(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    train_loader, val_loader, train_dataset = build_dataloaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = create_model(
        num_classes=len(train_dataset.classes),
        architecture=args.architecture,
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        print(
            f"Эпоха {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model=model,
                train_dataset=train_dataset,
                output_path=args.output,
                labels_path=args.labels_output,
                architecture=args.architecture,
                image_size=args.image_size,
                val_accuracy=best_val_accuracy,
            )
            print(f"Сохранена лучшая модель: {args.output}")

    print(f"Обучение завершено. Лучшая val accuracy: {best_val_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
