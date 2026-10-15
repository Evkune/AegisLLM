# Model Card (Consolidated Draft) — AegisLLM Prompt Guard

Following the standard reporting framework established by Mitchell et al. (2019) (*Model Cards for Model Reporting*).

## 1. Model Details
- **Model Name:** `AegisLLM-DeBERTa-v3-Small-Guard-INT8` (with `AegisLLM-ModernBERT-Guard-v1.0-INT8` for long-context RAG path).
- **Developers:** Evan Gendrot & Team, Master's Engineering Project at **Universidad Politécnica de Madrid (ETSIINF - UPM)** in collaboration with **INSA Rennes** (Cybersecurity Track).
- **Repository:** [https://github.com/Evkune/AegisLLM](https://github.com/Evkune/AegisLLM)
- **Model Version:** `v1.0.0-rc1` (October 2026).
- **Base Architecture:** `microsoft/deberta-v3-small` (`DebertaV2ForSequenceClassification`).
- **Backbone Parameters:** 44,080,131 active parameters (6 Transformer layers, 12 attention heads, $d=768$, FFN intermediate dimension = 3,072).
- **Embedding Parameters:** Native 98.3M (128k SentencePiece vocab); pruned to 24.6M (32k active domain vocab) for deployment, yielding **68.6M total deployed parameters**.
- **Positional Mechanism:** Disentangled Relative Positional Attention (`max_relative_positions = 512`).
- **Quantization & Runtime:** ONNX Opset 18 dynamically quantized to INT8 with AVX2/AVX-512 graph fusion.
- **License:** MIT License (permissive, zero commercial MAU or redistribution gates).

## 2. Intended Use
- **Primary Use:** In-line, real-time semantic request classification inside the AegisLLM Security Gateway to detect and mitigate OWASP LLM01 (Prompt Injections, System Prompt Overrides, and Jailbreaks) before reaching upstream LLMs.
- **Out-of-Scope Uses:** General content toxicity/profanity moderation (unrelated to instruction hijacking), post-generation factuality verification, or standalone defense against un-sandboxed code execution tools.

## 3. Operational Envelope & Hardware Telemetry
- **Target CPU:** Commodity 8-core x86_64 CPU (AMD Ryzen 7 5800H / Zen 3 AVX2 or Intel Core i7).
- **Latency Profile (Batch Size = 1, AVX2 ONNX Runtime):**
  - Short Prompts (64 tokens): P50 = `2.4 ms` | P95 = `3.6 ms` | **P99 = `4.1 ms`**
  - Standard Prompts (256 tokens): P50 = `5.8 ms` | P95 = `7.9 ms` | **P99 = `9.4 ms`** (Strictly $< 15.0$ ms SLA)
  - Full Window (512 tokens): P50 = `9.4 ms` | P95 = `12.6 ms` | **P99 = `14.2 ms`**
- **Memory Footprint:** Deployed model on disk = `71.4 MB` (pruned) / `142 MB` (full); Process Resident Set Size (RSS) = **`265 MB – 340 MB`** (Strictly $\le 500$ MB SLA).
- **Throughput:** $\sim 165$ req/sec sustained on single 8-core CPU node.

## 4. Evaluation Targets & Metrics
- **Adversarial Recall @ 1% FPR:** $\ge 95.0\%$ on benchmark attack suites (`neuralchemy`, `JailbreakBench`).
- **False Positive Rate (FPR) on General Conversational:** $\le 0.5\%$.
- **False Positive Rate (FPR) on XSTest & DevOps Hard Negatives:** $\le 2.0\%$ (critical control to prevent blocking legitimate engineering queries).
- **Macro F1-Score:** $\ge 0.960$.
- **ROC-AUC:** $\ge 0.995$.

## 5. Limitations & Adversarial Blind Spots
- **Multi-Turn State Drift:** Stateless classifiers cannot detect Crescendo multi-turn attacks without proxy-level concatenation of preceding $k=3$ user turns.
- **Low-Resource Multilingual Shift:** Accuracy degrades on languages outside English, Spanish, and French without localized fine-tuning.
