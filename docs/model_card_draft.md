# Model Card (Preliminary Draft) — AegisLLM Detector

Following the reporting framework established by Mitchell et al. (2019) (*Model Cards for Model Reporting*).

## 1. Model Details
- **Model Name:** AegisLLM Prompt Guard Classifier
- **Model Type:** Bidirectional Transformer Encoder fine-tuned for sequence classification (Phase 1: Baseline Structural Engine; Phase 2: ModernBERT / DeBERTa-v3-small INT8 ONNX export).
- **Developers:** Evan Gendrot & Team (Master SE, UPM / INSA Rennes).
- **License:** MIT License.

## 2. Intended Use
- **Primary Intended Uses:** Real-time in-line inspection of incoming text prompts within enterprise AI gateways and API proxies to identify prompt injection attempts, system prompt overrides, and jailbreak vectors.
- **Out-of-Scope Uses:** Autonomous code evaluation, general sentiment analysis, toxic speech classification outside adversarial manipulation context, or post-generation factuality verification.

## 3. Factors and Operational Envelope
- **Hardware Envelope:** x86_64 and ARM64 standard CPU nodes.
- **Latency Target:** Sub-15ms P99 per prompt under sequential evaluation.
- **Context Length:** 512 to 8,192 tokens depending on the underlying encoder backbone.

## 4. Evaluation Targets & Metrics
- **Recall on Injections:** >= 95% on benchmark attack suites.
- **False Positive Rate (FPR) on Benign Prompts:** <= 2.0% on XSTest / enterprise conversational corpora to prevent business disruption.
- **Inference Latency:** Mean < 5 ms, P99 < 15 ms on CPU.

## 5. Ethical Considerations & Limitations
- Detection relies on semantic and syntactic indicators; evolving zero-day jailbreaks requiring multi-turn conversational priming remain an ongoing defense-in-depth challenge.
- Continuous active threat intelligence updates and rule re-training are planned to counteract model drift.
