# AgroCrimea AI

Учебный проект по дисциплине «Интеллектуальные системы»: веб-приложение для диагностики заболеваний листьев томата по изображению.

Система объединяет три интеллектуальных компонента:

1. **Чужая модель ИИ** — внешний детектор на базе YOLOv8 через `ultralytics`. Он демонстрирует использование готовой предобученной модели компьютерного зрения и проверяет, что изображение похоже на растение или лист. Если YOLOv8 недоступна, используется резервная проверка качества изображения.
2. **Своя модель** — PyTorch-классификатор заболеваний листьев томата. В проекте реализовано обучение transfer learning на `mobilenet_v3_small` или `efficientnet_b0`.
3. **Экспертная система** — продукционная система правил «ЕСЛИ — ТО». Она принимает факты от внешней модели, собственной модели и пользователя, затем формирует итоговый статус, уровень риска, объяснение и рекомендации.

## Структура проекта

```text
agrocrimea-ai/
  app.py
  requirements.txt
  README.md
  src/
    __init__.py
    external_detector.py
    disease_model.py
    expert_system.py
    recommendations.py
    preprocessing.py
    utils.py
  training/
    train.py
    evaluate.py
  models/
    labels.json
    disease_model.pt          # создается после обучения
    confusion_matrix.png      # создается после оценки
  data/
    rules.json
    recommendations.json
  sample_images/
    README.md
```

## Установка

Рекомендуется использовать Python 3.10 или новее.

```bash
cd agrocrimea-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

На Linux или macOS активация окружения будет другой:

```bash
source .venv/bin/activate
```

## Подготовка датасета

Датасет не скачивается автоматически. Его нужно подготовить вручную и положить в папку `dataset` внутри проекта.

Ожидаемая структура:

```text
dataset/
  train/
    healthy/
    early_blight/
    late_blight/
  val/
    healthy/
    early_blight/
    late_blight/
  test/
    healthy/
    early_blight/
    late_blight/
```

Названия классов должны совпадать с папками:

- `healthy` — здоровый лист;
- `early_blight` — альтернариоз;
- `late_blight` — фитофтороз.

Если в скачанном архиве папки называются с суффиксом, например `healthy227`, `Early_blight227`, `Late_blight227`, в рабочем датасете их лучше переименовать или скопировать в чистые имена:

```text
healthy227      -> healthy
Early_blight227 -> early_blight
Late_blight227  -> late_blight
```

## Обучение модели

Пример запуска на 5 эпох:

```bash
python training/train.py --data dataset --epochs 5
```

Скрипт:

- загружает изображения через `torchvision.datasets.ImageFolder`;
- применяет `Resize`, `RandomHorizontalFlip`, `RandomRotation`, `ToTensor`, `Normalize`;
- использует CPU или CUDA, если CUDA доступна;
- выводит `loss` и `accuracy` по эпохам;
- сохраняет веса в `models/disease_model.pt`;
- сохраняет соответствие классов в `models/labels.json`.

Можно выбрать архитектуру:

```bash
python training/train.py --data dataset --epochs 5 --architecture efficientnet_b0
```

Если загрузка предобученных весов torchvision недоступна, скрипт создаст модель без них и выведет предупреждение.

## Оценка модели

После обучения можно проверить модель на тестовой выборке:

```bash
python training/evaluate.py --data dataset --split test
```

Скрипт выводит:

- accuracy;
- precision;
- recall;
- f1-score;
- classification report;
- confusion matrix.

Изображение матрицы ошибок сохраняется в `models/confusion_matrix.png`.

## Запуск Streamlit-приложения

```bash
streamlit run app.py
```

Если `models/disease_model.pt` еще не создан, приложение не падает. Оно покажет сообщение:

```text
Модель заболевания не найдена. Сначала обучите модель командой python training/train.py --data dataset
```

Это нормальное состояние до обучения собственной модели.

## Как работает приложение

1. Пользователь загружает изображение листа томата.
2. Внешний детектор `src/external_detector.py` пытается найти растение или лист через YOLOv8. Если YOLOv8 не дала надежного результата, запускается резервная проверка размера, яркости и цветового разнообразия изображения.
3. Собственная модель `src/disease_model.py` классифицирует изображение по классам `healthy`, `early_blight`, `late_blight`.
4. Экспертная система `src/expert_system.py` читает правила из `data/rules.json` и применяет их к фактам: наличие объекта, уверенность модели, болезнь, регион, влажность и сезон.
5. Модуль `src/recommendations.py` получает рекомендации из `data/recommendations.json`.
6. `app.py` показывает человеку понятный итог: диагноз, уверенность, риск, сработавшие правила, объяснение и рекомендации.

## Правила экспертной системы

Правила хранятся в `data/rules.json`. Примеры:

- `R1`: если лист растения не обнаружен, итоговый статус `no_object`;
- `R5`: если уверенность модели не ниже `0.7` и класс не `healthy`, заболевание считается диагностированным;
- `R6`: если обнаружен фитофтороз и влажность высокая, риск повышается до высокого;
- `R8`: если модель уверенно определила `healthy`, итоговый статус `no_disease`.

## Сценарий демонстрации для защиты

1. Показать структуру проекта и JSON-файлы правил.
2. Объяснить, что датасет кладется в папку `dataset` и не скачивается автоматически.
3. Запустить обучение:

```bash
python training/train.py --data dataset --epochs 5
```

4. Запустить оценку:

```bash
python training/evaluate.py --data dataset --split test
```

5. Запустить интерфейс:

```bash
streamlit run app.py
```

6. Загрузить фото листа, выбрать регион Крыма, влажность и сезон.
7. Показать три этапа рассуждения: внешний детектор, PyTorch-модель, экспертная система.
8. Показать список сработавших правил и рекомендации.

## Файлы для проверки реализации

- `app.py` — Streamlit-интерфейс;
- `src/external_detector.py` — внешняя модель YOLOv8 и fallback-проверка;
- `src/disease_model.py` — загрузка и инференс собственной PyTorch-модели;
- `src/expert_system.py` — интерпретатор продукционных правил;
- `src/recommendations.py` — рекомендации по диагнозу и риску;
- `training/train.py` — обучение модели;
- `training/evaluate.py` — оценка модели;
- `data/rules.json` — база правил;
- `data/recommendations.json` — база рекомендаций.
