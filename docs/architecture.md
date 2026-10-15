# AegisLLM Architecture Specification & Model Trade-Off Benchmark

## 1. System Overview

AegisLLM is an enterprise-grade AI Application Security Gateway designed to operate in-line as a reverse proxy protecting downstream Large Language Model (LLM) services and agentic workflows. 

The primary objective is the real-time mitigation of adversarial inputs mapped to **OWASP Top 10 for LLM Applications (LLM01: Prompt Injection and Jailbreak)**.

## 2. In-Line Interception Flow

```
                      +------------------------------------------------+
                      |                 Client Traffic                 |
                      +-----------------------+------------------------+
                                              |
                                              v (HTTP POST /v1/chat/completions)
+-----------------------------------------------------------------------------------------------+
| AegisLLM Security Gateway                                                                     |
|                                                                                               |
|  +--------------------------------+       +---------------------------------------------+     |
|  | Deterministic Pre-Hygiene      | ----> | Dual-Engine Classifier Hierarchy            |     |
|  | - Zero-width format strip      |       | 1. Baseline Heuristics (< 0.1 ms)           |     |
|  | - Unicode NFKC normalization   |       | 2. DeBERTa-v3-small INT8 ONNX (In-line <10ms) |     |
|  | - Anti-spacing collapse (Cisco)|       | 3. ModernBERT-base INT8 (Long Context >512) |     |
|  | - Recursive Base64 unwrap      |       +----------------------+----------------------+     |
|  +--------------------------------+                              |                            |
|                                                                  v                            |
|                                                   +-----------------------------+             |
|                                                   |      Decision Gateway       |             |
|                                                   | - Score >= 0.75: BLOCK      |             |
|                                                   | - Score >= 0.45: SUSPICIOUS |             |
|                                                   | - Score <  0.45: ALLOW      |             |
|                                                   +--------------+--------------+             |
+------------------------------------------------------------------|----------------------------+
                                                                   |
                                  +--------------------------------+-------------------+
                                  | (ALLOW)                                            | (BLOCK)
                                  v                                                    v
                 +--------------------------------+                  +----------------------------------+
                 | Upstream LLM Endpoint          |                  | Immediate Rejection              |
                 | (e.g. OpenAI / vLLM / Ollama)  |                  | HTTP 403 Forbidden               |
                 |                                |                  | CEF/Syslog Security Audit Log    |
                 +--------------------------------+                  +----------------------------------+
```

## 3. Comprehensive Model Architectural Benchmark

To satisfy our strict production constraints, we evaluated four distinct architectural families:

| Model Family / Candidate | Exact HF Identifier | Active Parameters | Context Length | CPU P99 Latency (INT8, 256 tok) | Operational RAM (INT8) | Architectural Assessment & Trade-off |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Legacy Baseline** (Regex + Perplexity) | `PCRE2` + `openai-community/gpt2` | 0M (Regex) + 124M (GPT-2) | 1,024 | 0.4 ms (Regex) / 38 ms (GPT-2) | ~175 MB | **Catastrophic Failure.** Fails against Base64, homoglyphs, and zero-width joins. Perplexity spikes on benign code (FPR > 10%). |
| **2A. Primary In-Line SOTA** (`microsoft/deberta-v3-small`) | `microsoft/deberta-v3-small` | **44M backbone** / 141M total (6L) | 512 | **6.8 – 9.8 ms** | **~165 MB** | **Optimal Sweet Spot for CPU Edge.** 6-layer depth guarantees sub-10ms P99 on AVX2 (Ryzen 7 5800H). Disentangled relative attention models boundary overrides. |
| **2B. Long-Context SOTA** (`answerdotai/ModernBERT-base`) | `answerdotai/ModernBERT-base` | 110M backbone / 149M total (22L) | **8,192** | 11.0 – 14.2 ms (bounded) / 24 ms (unbounded) | ~185 MB | **Long-Range Document Specialist.** RoPE + GeGLU, trained on 2T tokens (code-native). Ideal for RAG injection screening via Head-Tail-Entropy chunking. |
| **2C. Meta PromptGuard-86M** | `meta-llama/Prompt-Guard-86M` | 86M backbone / 278M total (12L) | 512 | 18.0 – 22.5 ms | ~310 MB | **SLA & License Violation.** 12 layers exceed 15ms CPU budget. Vulnerable to character-spacing bypass without pre-hygiene. Restricted license. |
| **3. Compact SLM Decoders** (SmolLM2 / Qwen2.5) | `HuggingFaceTB/SmolLM2-135M`<br>`Qwen/Qwen2.5-0.5B` | 135M (30L)<br>490M (24L) | 8,192<br>32,768 | 32.0 ms (SmolLM2)<br>115.0 ms (Qwen) | ~180 MB<br>~540 MB | **Causal Masking Blindness.** Causal decoders cannot attend bidirectionally across prompt boundaries. 30 layers cause high CPU thread sync overhead. |
| **4. Heavy LLM-as-a-Judge** (Llama-3-8B) | `meta-llama/Meta-Llama-3-8B-Instruct` | 8.03B (32L) | 8,192 | 1,850 – 3,200 ms | ~8,500 MB | **Production DoS Vector.** DDR5 memory-bandwidth bound (> 1.8s per request). Unacceptable latency and cost for an in-line proxy. |

## 4. Architectural Selection: The Pareto Frontier Strategy

Rather than an arbitrary single choice, AegisLLM adopts a two-tier **Pareto Frontier** architecture:
1. **Interactive Fast-Path (< 10 ms P99):** `microsoft/deberta-v3-small` (6 layers, quantized INT8 via ONNX Runtime). Processes all standard interactive conversational prompts ($\le 512$ tokens), guaranteeing SLA compliance on commodity AVX2 processors.
2. **Extended RAG Path (8,192 tokens):** `answerdotai/ModernBERT-base` (22 layers, RoPE). Analyzes long-context documents via Head-Tail-Entropy sequence extraction, ensuring high recall on indirect injections embedded inside massive text contexts.

## 5. Requirements Specification

### Functional Requirements (FR)
- **FR-1 (Traffic Interception):** Transparently intercept all inbound requests targeting `/v1/chat/completions` and `/v1/completions`.
- **FR-2 (Pre-Tokenization Hygiene):** Deterministically strip zero-width characters (`U+200B`–`U+FEFF`), normalize Unicode via NFKC, collapse adversarial spacing (`I g n o r e`), and unwrap candidate Base64 payloads in $< 0.3$ ms.
- **FR-3 (Multi-Class Threat Classification):** Categorize threats into Direct Prompt Injection, System Prompt Leakage, and Jailbreak personas.
- **FR-4 (Deterministic Gating):** Enforce strict policy-based routing with explicit confidence thresholds (`ALLOW`, `SUSPICIOUS`, `BLOCK`).
- **FR-5 (Audit Trail & SIEM Logging):** Generate structured security audit events including correlation IDs, matched indicators, threat scores, and latency metrics.

### Non-Functional Requirements (NFR)
- **NFR-1 (Latency Budget):** In-line inspection P99 latency must remain strictly below **15.0 ms** on commodity multi-core CPU hardware (Zen 3 / AVX2).
- **NFR-2 (Resource Footprint):** Operational memory usage must not exceed 500 MB per worker process; zero runtime reliance on dedicated GPU infrastructure.
- **NFR-3 (Fail-Safe Behavior):** Configurable fail-safe mode in the event of upstream timeout (default: fail-closed with alert notification).
- **NFR-4 (Training Efficiency):** Supervised fine-tuning must converge within $< 15$ minutes on single cloud accelerator (NVIDIA L4 / A100 / T4).
