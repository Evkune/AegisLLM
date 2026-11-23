# Model Card — AegisLLM Prompt Guard (`DeBERTa-v3-Small` ONNX INT8)

Following the standard reporting framework established by Mitchell et al. (2019), *Model Cards for Model Reporting*. Every metric below is **100% empirically measured** from the production artifact `models/deberta_v3_small_aegis_v1/model_int8_dynamic.onnx` and `reports/evaluation_summary.json`.

---

## 1. Model Details
- **Model Name:** `AegisLLM-DeBERTa-v3-Small-Guard-INT8` (`model_int8_dynamic.onnx`)
- **Developers:** Evan Gendrot & Amaury Dubuc — Master of Science in Software Engineering, **Universidad Politécnica de Madrid (ETSIINF — UPM)** / **INSA Rennes** (Cybersecurity Track).
- **Repository:** [https://github.com/Evkune/AegisLLM](https://github.com/Evkune/AegisLLM)
- **Model Version:** `v1.0.0` (Milestone 2 Production Release, November 2026)
- **Base Architecture:** `microsoft/deberta-v3-small` (`DebertaV2ForSequenceClassification`, 3 classes: `0 = BENIGN`, `1 = PROMPT_INJECTION`, `2 = JAILBREAK`).
- **Parameter Breakdown:**
  - **Transformer Backbone:** `44.1M` active parameters (6 layers, 12 heads, $d = 768$, FFN intermediate = `3,072`, disentangled relative positional attention).
  - **Token Embedding Table (Unpruned):** `98.4M` parameters (`128,100`-token SentencePiece vocabulary stored in FP32 = **`393.5 MB`** on disk).
  - **Total Deployed ONNX Size:** **`434.94 MB`** (`393.5 MB` FP32 embeddings + `~41.4 MB` INT8 dynamic Transformer weights).
- **Training Recipe:** Two-stage knowledge distillation from a fine-tuned `microsoft/deberta-v3-base` Teacher (`86.0M` backbone, 2 epochs FP32) using `HybridDistillationCSAFL` ($\alpha_{\text{distill}} = 0.40$, $T_{\text{distill}} = 2.0$) combining **Cost-Sensitive Asymmetric Focal Loss (CS-AFL)** (`class_weights = [1.0, 1.172, 3.012]`, $\gamma = [2.5, 1.5, 2.0]$, $\beta_{\text{fp}} = 2.25$, label smoothing `0.03`), **FGM adversarial embedding perturbations** ($\varepsilon = 0.50$), and **Layer-Wise Learning Rate Decay** (`llrd_decay = 0.90`).
- **Quantization & Calibration:** ONNX Opset 18 (`ORT_ENABLE_BASIC` pre-optimization) + **Dynamic INT8 quantization** (`QInt8` on `MatMul`/`Gemm`, excluding `embeddings`, `rel_embeddings`, `classifier`, `pooler`, `layernorm`, `_proj_1`), followed by **post-quantization temperature recalibration** strictly on Validation ($T_{\text{FP32}} = 0.6759 \rightarrow T_{\text{int8}} = 0.6609$) and operating threshold selection ($\tau_{95} = 0.9879$, $\tau_{1\%} = 0.4449$).
- **License:** MIT License.

---

## 2. Intended Use & Scope Boundaries
- **Primary Intended Use:** Inline, CPU-only semantic request classification inside the AegisLLM reverse proxy to detect and mitigate **OWASP LLM01 (Direct Prompt Injection, Indirect RAG/Tool Injection, and Adversarial Jailbreaks)** before requests reach upstream LLMs.
- **Intended Users:** AI platform engineers, DevSecOps teams, and security researchers deploying OpenAI-compatible gateways without dedicated GPU guardrails.
- **Explicit Out-of-Scope Uses:**
  - General content toxicity, hate speech, or direct policy-harm moderation unrelated to instruction hijacking (e.g., unwrapped harmful queries in `XSTest`/`JailbreakBench` belong to downstream content-safety classifiers such as Llama Guard).
  - Post-generation hallucination or factuality verification.
  - Standalone protection for un-sandboxed code execution or privileged tool calls (defense-in-depth least-privilege sandboxing remains mandatory).

---

## 3. Empirical Evaluation & Ablation Summary

Evaluated on the strictly zero-leakage in-domain test split (**`N = 4,438`**, zero MinHash LSH overlap with train/val) and three isolated OOD benchmarks (**`XSTest`**, **`NotInject`**, **`JailbreakBench`**, `N = 1,289`).

### 3.1 Production Model vs. Baselines & Quantization Duel

| Model Variant | Macro-F1 (95% CI) | ROC-AUC | PR-AUC | FPR @ 95% TPR | TPR @ 1% FPR | ECE | DevOps FPR | XSTest Safe FPR | NotInject FPR | JBB TPR | McNemar vs Tier-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tier-1 TF-IDF + LinearSVC | `0.918 ± 0.012` | `0.9867` | `0.9859` | `6.99%` | `85.74%` | `0.0316` | `8.33%` | `2.00%` | `23.60%` | `33.00%` | Reference ($H_0$) |
| Teacher (`deberta-v3-base`, 86M) | `0.933 ± 0.013` | `0.9983` | `0.9981` | `0.37%` | `98.55%` | `0.0106` | `1.39%` | `0.00%` | `21.24%` | `34.75%` | `p < 0.0001` |
| Student FP32 (`cs_afl` 3-seed mean ± std) | `0.934 ± 0.008` | `0.9991` | `0.9989` | `0.25% ± 0.04%` | `98.97%` | `0.0083` | `0.46%` | `0.00%` | `21.14% ± 7.86%` | `28.08%` | `p < 0.0001` |
| Student FP32 (`soup_fp32`, `seed=42`) | `0.934 ± 0.013` | `0.9990` | `0.9988` | `0.25%` | `99.00%` | `0.0088` | `1.39%` | `0.00%` | `14.45%` | `25.50%` | `p < 0.0001` |
| **Prod: ONNX INT8 Dynamic** | **`0.926 ± 0.013`** | **`0.9988`** | **`0.9986`** | **`0.29%`** | **`98.55%`** | **`0.0074`** | **`1.39%`** | **`0.00%`** | `26.55%` | `28.50%` | **`p < 0.0001`** |
| Challenger: ONNX INT8 Static QDQ | `0.906 ± 0.015` | `0.9982` | `0.9979` | `0.53%` | `97.86%` | `0.0089` | `2.78%` | `0.00%` | `35.69%` | `30.75%` | `p < 0.0001` |

- **Quantization Drift (`int8_dynamic` vs `soup_fp32` on Test `N = 4,438`):** Top-1 agreement **`98.76%`**, mean KL divergence **`0.0064`**, mean $|\Delta\text{logit}| = 0.1268$, $\Delta\text{FPR@95\%TPR} = +0.04\%$, $\Delta\text{PR-AUC} = -0.0002$. (By contrast, `int8_static_qdq` degrades to `95.92%` agreement and `0.0573` KL — `9×` higher drift.)
- **270-Vector OWASP Red-Team Suite (`reports/redteam_asr_report.json`):** **`0.00%` ASR (`210/210` attacks blocked)**, **`0.00%` FPR on UUID/MD5/SHA/JWT identifiers (`0/15`)**, **`3.33%` overall hard-negative FPR (`2/60`)**.

---

## 4. Operational Envelope & Hardware Telemetry

Measured on **AMD Ryzen 7 5800H** (8C/16T Zen 3 AVX2, no AVX-512 VNNI), `onnxruntime 1.32.0`, `CPUExecutionProvider` (`intra_op_num_threads = 8`, `inter_op_num_threads = 1`, `ORT_SEQUENTIAL`, `IOBinding`).

- **Isolated ONNX Runtime Latency (`B = 1`):**
  - `L = 64` tokens: P50 = `11.59 ms` | P95 = `12.44 ms` | **P99 = `13.59 ms`** (strictly `< 15.0 ms` SLA)
  - `L = 128` tokens: P50 = `22.51 ms` | P95 = `25.58 ms` | **P99 = `27.72 ms`**
  - `L = 256` tokens: P50 = `54.69 ms` | P95 = `57.19 ms` | **P99 = `59.66 ms`**
- **Live End-to-End FastAPI Proxy Load (`enforce` mode):**
  - `C = 1`: `90.9 req/s` | E2E P50 `12.35 ms` | Short Benign P99 **`15.33 ms`** | Peak RSS **`693.2 MB`**
  - `C = 16`: **`176.4 req/s`** (peak throughput) | E2E P50 `91.17 ms` | Peak RSS **`697.1 MB`**
  - `L = 4,096` tokens (bounded `Head + Tail + Top-2 Hot-Spots` windowing): E2E P99 **`127.76 ms`** with `100%` buried-attack detection.
- **Memory & Container Footprint (Transparent Trade-Off):**
  - **Docker Image (`Dockerfile`):** **`326 MB`** (`python:3.11-slim`, non-root UID `10001`, zero PyTorch/Transformers).
  - **Process RSS (`693–698 MB`):** Exceeds the original Milestone 1 target of `500 MB` because DeBERTa-v3's unpruned `128,100`-token FP32 embedding table alone occupies **`393.5 MB`** (`~90.5%` of the `434.9 MB` ONNX file), plus the ORT CPU memory arena and FastAPI/Rust tokenizer runtime.

---

## 5. Known Limitations, Failure Modes & Milestone 3 Roadmap

1. **Trigger-Word Lexical Bias (`NotInject` OOD):** `int8_dynamic` incurs `26.55%` FPR on benign sentences intentionally stuffed with uppercase override tokens (`POLICY`, `INSTRUCTIONS`, `Disregarded`) — up from `14.45%` in FP32. **Mitigation:** Quantization-Aware Training (QAT) + trigger-word hard-negative augmentation in Milestone 3.
2. **Residual DevOps Shell False Positives:** `2/20` DevOps red-team prompts containing sensitive shell/credential strings (`chmod 600 ~/.ssh/id_ed25519`, `curl -v https://api.internal/healthz`) exceed $\tau_{95}$.
3. **Construct Boundary on Direct Harmful Prompts (`JBB`):** Unwrapped policy-harmful prompts without adversarial override framing yield `28.50%` TPR at $\tau_{95}$ (`40.00%` at `0.50`); pair AegisLLM with a downstream content-moderation guard for non-LLM01 harm filtering.
4. **Unpruned Embedding Table RSS (`693 MB`):** Pruning the SentencePiece vocabulary from `128k → 32k` domain tokens in Milestone 3 will shrink the embedding table from `393.5 MB` to `~98 MB`, cutting total ONNX size to `~140 MB` and live RSS to `< 350 MB`.
5. **Multilingual & Multi-Turn Edge Cases:** Non-English theatrical roleplay (e.g., German framing) and slow multi-turn crescendo priming beyond the `k = 3` user-turn window remain active expansion axes.
