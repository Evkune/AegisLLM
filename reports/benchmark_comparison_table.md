# AegisLLM - Milestone 2 Empirical Benchmark (100% Measured)

| Model / Variant | Params | Macro F1 (95% CI) | PR-AUC | ROC-AUC | FPR @ 95% TPR | DevOps FPR | XSTest Safe FPR | NotInject FPR | JBB TPR | CPU P99 (ms) | McNemar p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Tier 1: TF-IDF + Calibrated LinearSVC** | 0.15M | 0.918 ± 0.012 | 0.986 | 0.987 | 7.0% | 8.3% | 2.0% | 23.6% | 33.0% | 14.2 ms | Reference (H0) |
| **DeBERTa-v3-Small (CE, seed=42)** | 44.1M | 0.940 ± 0.012 | 0.999 | 0.999 | 0.3% | 1.4% | 0.0% | 29.2% | 30.0% | N/A | p < 0.0001 |
| **DeBERTa-v3-Small (CS_AFL, No Distill, seed=42)** | 44.1M | 0.942 ± 0.012 | 0.999 | 0.999 | 0.3% | 1.4% | 0.0% | 31.0% | 28.7% | N/A | p < 0.0001 |
| **DeBERTa-v3-Small (CS_AFL, seed=1337)** | 44.1M | 0.941 ± 0.011 | 0.999 | 0.999 | 0.3% | 0.0% | 0.0% | 29.8% | 31.2% | N/A | p < 0.0001 |
| **DeBERTa-v3-Small (CS_AFL, seed=2026)** | 44.1M | 0.926 ± 0.013 | 0.999 | 0.999 | 0.2% | 0.0% | 0.0% | 19.2% | 27.5% | N/A | p < 0.0001 |
| **DeBERTa-v3-Small (CS_AFL, seed=42)** | 44.1M | 0.934 ± 0.013 | 0.999 | 0.999 | 0.2% | 1.4% | 0.0% | 14.5% | 25.5% | N/A | p < 0.0001 |
| **DeBERTa-v3-Small (FOCAL, seed=42)** | 44.1M | 0.933 ± 0.013 | 0.999 | 0.999 | 0.2% | 1.4% | 0.0% | 15.3% | 25.0% | N/A | p < 0.0001 |
| **Candidate: DeBERTa-v3-Small (ONNX INT8 Dynamic, Recalibrated)** | 44.1M (INT8) | 0.926 ± 0.013 | 0.999 | 0.999 | 0.3% | 1.4% | 0.0% | 26.5% | 28.5% | 13.6 ms | p < 0.0001 |
| **Challenger: DeBERTa-v3-Small (ONNX INT8 Static QDQ, Recalibrated)** | 44.1M (INT8) | 0.906 ± 0.015 | 0.998 | 0.998 | 0.5% | 2.8% | 0.0% | 35.7% | 30.8% | 16.5 ms | p < 0.0001 |
| **Candidate: DeBERTa-v3-Small (CS-AFL + FGM + Greedy Soup FP32)** | 44.1M | 0.934 ± 0.013 | 0.999 | 0.999 | 0.2% | 1.4% | 0.0% | 14.5% | 25.5% | N/A | p < 0.0001 |
| **Teacher: DeBERTa-v3-Base (Fine-Tuned FP32)** | 86.0M | 0.933 ± 0.013 | 0.998 | 0.998 | 0.4% | 1.4% | 0.0% | 21.2% | 34.8% | N/A | p < 0.0001 |
