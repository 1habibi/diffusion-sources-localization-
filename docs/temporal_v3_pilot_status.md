# Temporal-v3 paired pilot: implementation status

2026-09-28: реализован отдельный no-training pipeline для source-blind раннего снимка после шага 1 и неизменного финального входа S1b.

- Notebook: `notebooks/colab_temporal_v3_pilot.ipynb`, семь отдельных кодовых ячеек; не Run all.
- CLI: `python -m diffusion_sources.temporal_pilot_cli {smoke,select,validate,confirm}` с `--data-dir`, `--run-dir`, `--output-dir`, `--device`.
- Beta выбирается на первых 540 train-примерах; validation не участвует в настройке.
- Контрольный F1 проверяется против frozen metrics; test/holdout не читаются.
- Новый output: `reports/runs/temporal_v3_paired_pilot/v1/`; старые архивы, checkpoint и отчёты не перезаписываются. При изменении code/input hashes выбирать новый output suffix; частичные стадии не удаляются автоматически.
- Stage manifests содержат code/input hashes, RNG policy, train indices, beta и runtime metadata; paired CSV, metrics и CI сохраняются отдельно.
- Первичный gate: delta F1 >=0.02, CI lower >0, без падения k=2/3 более0.02. При pass — frozen7027/7028 с теми же beta/масками, без обучения.

Проверки до финального review: 175 passed, два существующих PyTorch deprecation warnings; compileall и git diff --check прошли. Реальный replay шести Facebook train-каскадов совпал, около0.02 секунды после загрузки архива. Synthetic integration прогнал inference/сохранение/resume без обучения и с неизменными хешами входов.

Frozen S1b/validation F1 здесь не оценивался: checkpoint находится на Drive. Colab-пилот ещё не запускался, никаких результатов улучшения v3 не заявляется. Изменения пока в отдельной ветке; публикация для git pull требует отдельного согласования.

## Финальная проверка

Независимый whole-branch review нашёл два важных замечания, оба закрыты одним исправляющим проходом:

- Frozen distance cache теперь только читается, его SHA256 (либо отсутствие) включён в identity стадий. При отсутствии cache расстояния пересчитываются в памяти с явным сообщением, файл по старому пути не создаётся. Оба дефекта воспроизведены RED-тестами и исправлены GREEN.
- Добавлен полный select → validate → confirm путь с контролируемыми frozen inference records, реальными scoring/metrics/artifacts; проверены locked beta, чтение только train при selection, отсутствие retuning, resume, mismatch saved F1, отказ при failed primary gate и изменённых early masks. Чувствительность baseline regression проверена отдельным процессом с намеренно отключённой контрольной проверкой: RED; штатная реализация GREEN.

Итоговый полный набор: **181 passed**, два прежних предупреждения PyTorch; compileall и diff check прошли. Критических или отложенных minor-замечаний нет. Реальный Drive/GPU pilot остаётся не выполнен.

## Решения исполнения

1. Native worktree tool не распознал родительскую папку как Git repository; использован Git worktree в `.worktrees/`. Цена: нет автоматической привязки worktree в приложении.
2. Bash ledger scripts не запускались из-за Windows signal-pipe permissions; bookkeeping выполнен PowerShell/apply_patch с теми же test gates. Цена: ручной учёт.
3. Вместо текстовых notebook assertions проверяется исполнение ячеек с заменой внешних subprocess/Colab-зависимостей. Цена: реальную Colab/GPU-среду нужно проверить отдельно.
4. Исходные журналы имели незакоммиченные записи пользователя, отсутствующие в worktree; статус дописан там, а этот независимый документ сохранён в feature branch. Цена: журналы ещё требуют отдельного коммита при интеграции.
