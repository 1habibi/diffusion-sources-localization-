# v1 baseline — обучение основных моделей в Google Colab

Этот документ и `notebooks/colab_training.ipynb` относятся к завершенной неизменяемой версии `v1_baseline`. Для `snapshot-v2` создан отдельный `notebooks/colab_training_snapshot_v2.ipynb`; существующий сценарий не удаляется и не переиспользуется для записи новых результатов.

Notebook `snapshot-v2` выполняет GPU smoke, последовательные S0-S7 от фактического validation-победителя, ограниченную S5-сетку, S7 safety gates, leave-one-component-out проверки, freeze manifest и финальные training seeds. Он принудительно сохраняет `evaluate_test: false`; существующий test и запечатанный final holdout открываются позже отдельной процедурой после переноса и аудита всех validation-решений.

Основная логика остается в репозитории. Colab используется только как GPU-исполнитель, Google Drive хранит датасет и результаты.

## Структура Google Drive

```text
MyDrive/diffusion-sources/
├── data/
│   └── facebook_main/
│       ├── graph.npz
│       ├── train.npz
│       ├── validation.npz
│       └── test.npz
└── reports/
```

Каталог `data/generated/facebook_main`, созданный локально, нужно загрузить в `MyDrive/diffusion-sources/data/facebook_main`.

## Порядок запуска

1. Выбрать GPU runtime.
2. Подключить Drive.
3. Клонировать `https://github.com/1habibi/diffusion-sources-localization-.git`.
4. Проверить `torch.cuda.is_available()`.
5. Установить проект без замены предустановленного CUDA-PyTorch.
6. Создать runtime-конфигурацию с абсолютными путями Drive.
7. Выполнить короткий запуск на ограниченном split.
8. Запустить один полный seed.
9. Проверить `metrics.json`, VRAM и время эпохи.
10. Запустить остальные модели и seed по отдельности.

## Возобновление

В конфигурациях установлено `resume: true`. После каждой эпохи атомарно сохраняется `last_checkpoint.pt`, содержащий:

- текущую эпоху;
- model state;
- optimizer state;
- лучший checkpoint и validation score;
- счетчик early stopping;
- историю train/validation;
- состояния Python, NumPy, Torch и CUDA RNG.

Повторный запуск той же команды в тот же output-каталог продолжает обучение со следующей эпохи. `best_model.pt` остается моделью, выбранной только по validation.

## Рекомендуемые каталоги результатов

```text
reports/
├── joint_full/seed_7026/
├── joint_full/seed_7027/
├── joint_full/seed_7028/
├── no_consistency/seed_7026/
└── node_only/seed_7026/
```

Разные модели и seed нельзя направлять в один каталог.

## Установка в Colab

Не следует выполнять обычный `pip install -e '.[dev]'`, пока не проверена совместимость требования Torch: pip может заменить CUDA-сборку. Notebook устанавливает обычные зависимости отдельно, затем проект с `--no-deps`.

После установки обязательно проверить:

```python
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
```

## Выбор batch size

Начать с `batch_size: 3`. Затем провести короткие запуски с 4, 6 и 8. Выбирается максимальное стабильное значение с запасом VRAM. В `metrics.json` автоматически сохраняются GPU, CUDA, общий объем VRAM и peak allocated memory.

## Основные команды

```bash
python scripts/train_model.py --config /content/train_joint.yaml \
  --output /content/drive/MyDrive/diffusion-sources/reports/joint_full/seed_7026

python scripts/train_node_model.py --config /content/train_node.yaml \
  --output /content/drive/MyDrive/diffusion-sources/reports/node_only/seed_7026
```

Тонкая интерактивная оболочка находится в `notebooks/colab_training.ipynb`.
