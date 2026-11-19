# AegisLLM - Milestone 2 Empirical Benchmark (100% Measured)

| Model / Variant | Params | Macro F1 (95% CI) | PR-AUC | ROC-AUC | FPR @ 95% TPR | DevOps FPR | XSTest Safe FPR | NotInject FPR | JBB TPR | CPU P99 (ms) | McNemar p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Tier 1: TF-IDF + Calibrated LinearSVC** | 0.15M | 0.918 ± 0.012 | 0.986 | 0.987 | 7.0% | 8.3% | 2.0% | 23.6% | 33.0% | 14.2 ms | Reference (H0) |
