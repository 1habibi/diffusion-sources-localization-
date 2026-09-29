# Материалы для дипломного отчёта

Все материалы для переноса в Word собраны здесь; экспериментальные notebook,
модели и протоколы остаются в прежних папках проекта.

| Папка | Содержимое |
|---|---|
| `text/` | [Текст, таблицы, подписи и выводы](text/temporal_v3_report.md); [формулы для Word](text/temporal_v3_report_formulas.md) |
| `notebooks/` | [Python notebook с шестью графиками](notebooks/temporal_v3_report_figures.ipynb) |
| `diagrams/` | [Две схемы Mermaid с пояснениями](diagrams/temporal_v3_report_diagrams.md) |
| `data/` | Неизменённый итоговый `independent_summary.json` и сведения об источнике `provenance.json` |
| `figures/` | Экспортированные PNG (300 dpi) и SVG |

## Как получить графики

1. Открой `notebooks/temporal_v3_report_figures.ipynb` в Colab. GPU не нужен.
2. Если Drive ещё не смонтирован, выполни отдельную ячейку:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Выполни все Python-ячейки notebook сверху вниз.
4. В Colab рисунки появятся в
   `/content/drive/MyDrive/diffusion-sources/thesis_report/figures/`.
   Локально — в `thesis_report/figures/` этого репозитория.

Notebook только читает уже готовый итоговый JSON. Не выполняются обучение,
генерация данных, открытие holdout или новый inference. Одноимённые рисунки
при повторном запуске заменяются; исходный JSON не меняется. SHA-256 исходного
JSON проверяется перед построением рисунков.

Для локального запуска нужны `numpy` и `matplotlib`; notebook автоматически
ищет папку пакета среди родительских каталогов. Можно явно задать `SOURCE_PATH`
и `OUTPUT_DIR` до запуска ячеек (импортировав `Path` из `pathlib`).

## Как переносить в Word

- Текст и пояснения — обычным копированием из `.md`, оформление выполняется в Word.
- Формулы — строки из LaTeX-блоков в поле формулы Word (Alt + =, режим LaTeX).
  Одиночные символы в пояснениях написаны обычным текстом.
- Графики — PNG или SVG из `figures/`, с подписями из текстового отчёта.
- Схемы — исходники Mermaid; перед вставкой в Word экспортировать в изображение.

Числа относятся к независимой оценке 1998 новых IC-каскадов прежнего
Facebook-графа, а не к исследовательскому validation и не к переносу на новую
топологию. Три checkpoints оценивают те же 1998 каскадов, не 5994 независимых.
