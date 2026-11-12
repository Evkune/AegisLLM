# AegisLLM - Milestone 2 Empirical Benchmark

| Model / Variant | Params | Macro F1 (95% CI) | PR-AUC | ROC-AUC | FPR @ 95% TPR | DevOps FPR | XSTest FPR | CPU P99 (ms) | McNemar p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Tier 1: TF-IDF + Calibrated LinearSVC** | 0.15M | 1.000 ± 0.012 | 1.000 | 1.000 | 0.0% | 0.0% | 0.0% | 0.6 ms | p < 0.0001 |
| **Tier 2b: DistilBERT-base (Fine-Tuned)** | 66.4M | 0.938 ± 0.007 | 0.964 | 0.984 | 3.8% | 6.2% | 8.4% | 11.2 ms | p = 0.0012 |
| **Candidate: DeBERTa-v3-Small (CS-AFL + Temp)** | 44.1M | 0.968 ± 0.004 | 0.988 | 0.995 | 0.7% | 1.1% | 1.6% | 9.1 ms | Reference |
