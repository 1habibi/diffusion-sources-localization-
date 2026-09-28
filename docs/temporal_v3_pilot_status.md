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
