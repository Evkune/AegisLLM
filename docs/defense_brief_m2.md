# AegisLLM — Milestone 2 Scientific Defense Brief

> **Reading Time:** ~4 minutes | **Scope:** Full Empirical Protocol, Ablation Matrix, Multi-Seed Stability, Systems Physics, Threats to Validity & Jury Q&A  
> **Primary Evidence:** `reports/evaluation_summary.json`, `reports/benchmark_comparison_table.md`, `reports/error_analysis_m2.md`, `reports/cpu_onnx_profile.md`, `reports/load_test_results.json`, `reports/latency_length_curve.json`, `reports/redteam_asr_report.json`

---

## 1. Executive Pitch (3 Sentences)

**AegisLLM** is an inline, CPU-only AI Application Security Gateway deployed as an OpenAI-compatible FastAPI reverse proxy that mitigates **OWASP LLM01 (Prompt Injection & Jailbreak)** with zero GPU dependency and zero PyTorch/Transformers imports in production. It combines a single-source-of-truth adversarial **Tier-0** pipeline (Unicode/homoglyph/encoding canonicalizer with structured UUID/SHA/JWT neural masking + ReDoS-proof Aho-Corasick prefilter) with a knowledge-distilled, adversarially trained **DeBERTa-v3-Small ONNX INT8** classifier (`44.1M` backbone). Across `N = 4,438` zero-leakage test prompts, three isolated OOD suites (`N = 1,289`), and a 270-vector OWASP red-team harness, the deployed ONNX INT8 model achieves **`0.29%` FPR @ 95% TPR** (`24×` better than Tier-1 TF-IDF), **`0.00%` red-team ASR**, **`0.00%` XSTest safe over-refusal**, and **`13.59 ms` P99 AVX2 CPU latency** (`L = 64`) inside a **`326 MB`** non-root Docker container.

---

## 2. Experimental Protocol & Architectural Decisions

1. **Zero-Leakage Corpus Hygiene (`data/manifest.json`, `tests/test_zero_leakage.py`):**
   - Aggregated 6 public, non-gated Hugging Face datasets into **`29,755` unique prompts** (`20,878` train / `4,439` val / `4,438` test).
   - Purged **`602`** near-duplicate cross-contaminants against `XSTest` (`450`), `NotInject` (`339`), and `JailbreakBench` (`500`) via MinHash LSH (`5`-gram shingles, `128` perms, `Jaccard >= 0.75`), then partitioned **`23,712`** structural clusters via `StratifiedGroupKFold(n_splits=20)`.
   - Zero shared clusters, zero exact overlap, zero OOD leakage enforced by automated pytest gates.
2. **Two-Stage Knowledge Distillation & Adversarial Training (`scripts/colab_train_pipeline.py`):**
   - **Stage 1 Teacher:** `microsoft/deberta-v3-base` (`86.0M`, FP32, 2 epochs, CS-AFL + LLRD) caching soft train logits (`float16`) for dark-knowledge distillation.
   - **Stage 2 Student:** `microsoft/deberta-v3-small` (`44.1M`, FP32, 3 epochs, length-bucketed batching, LLRD `0.90`, FGM embedding perturbations $\varepsilon = 0.50$, `HybridDistillationCSAFL` with $\alpha_{\text{distill}} = 0.40, T = 2.0$, $\beta_{\text{fp}} = 2.25$, $\gamma = [2.5, 1.5, 2.0]$).
   - **Greedy Model Soup:** Evaluated intermediate checkpoints on validation loss; retained `checkpoint-1564` (best validation loss).
3. **Quantization Duel & Post-Quantization Recalibration:**
   - Exported ONNX Opset 18 (`ORT_ENABLE_BASIC`), protecting sensitive disentangled-attention layers (`embeddings`, `rel_embeddings`, `classifier`, `pooler`, `layernorm`, `_proj_1`).
   - Benchmarked **`int8_dynamic`** vs **`int8_static_qdq`** on both numerical parity (`N = 4,438`) and real AVX2 Zen 3 latency (`AMD Ryzen 7 5800H`), followed by **post-quantization temperature scaling** strictly on Validation ($T_{\text{FP32}} = 0.6759 \rightarrow T_{\text{int8}} = 0.6609$, $\tau_{95} = 0.9879$).

---

## 3. Complete Empirical Results & Ablation Synthesis

### 3.1 Full Model & Ablation Matrix (`N = 4,438` Test + `N = 1,289` OOD)

| Experiment / Variant | Macro-F1 (95% CI) | ROC-AUC | FPR @ 95% TPR | TPR @ 1% FPR | ECE | Brier | DevOps FPR | XSTest Safe | NotInject FPR | JBB TPR | McNemar vs Tier-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tier-1 TF-IDF + LinearSVC | `0.918 ± 0.012` | `0.9867` | `6.99%` | `85.74%` | `0.0316` | `0.0427` | `8.33%` | `2.00%` | `23.60%` | `33.00%` | Ref ($H_0$) |
| Teacher (`deberta-v3-base`, 86M) | `0.933 ± 0.013` | `0.9983` | `0.37%` | `98.55%` | `0.0106` | `0.0107` | `1.39%` | `0.00%` | `21.24%` | `34.75%` | $\chi^2=83.3, p<10^{-4}$ |
| Loss Ablation: `CE` (`seed=42`) | `0.940 ± 0.012` | `0.9990` | `0.33%` | `99.05%` | `0.0086` | `0.0090` | `1.39%` | `0.00%` | `29.20%` | `30.00%` | $\chi^2=104.0, p<10^{-4}$ |
| Loss Ablation: `Focal` (`seed=42`) | `0.933 ± 0.013` | `0.9991` | `0.25%` | `99.05%` | `0.0083` | `0.0087` | `1.39%` | `0.00%` | `15.34%` | `25.00%` | $\chi^2=95.1, p<10^{-4}$ |
| Distill Ablation: `CS-AFL No Distill` (`s42`) | `0.942 ± 0.012` | `0.9989` | `0.29%` | `98.85%` | `0.0074` | `0.0095` | `1.39%` | `0.00%` | `30.97%` | `28.75%` | $\chi^2=105.3, p<10^{-4}$ |
| `CS-AFL + Distill` (`seed=42`) | `0.934 ± 0.013` | `0.9990` | `0.25%` | `99.00%` | `0.0088` | `0.0087` | `1.39%` | `0.00%` | `14.45%` | `25.50%` | $\chi^2=102.4, p<10^{-4}$ |
| `CS-AFL + Distill` (`seed=1337`) | `0.941 ± 0.011` | `0.9992` | `0.29%` | `99.05%` | `0.0079` | `0.0083` | `0.00%` | `0.00%` | `29.79%` | `31.25%` | $\chi^2=121.5, p<10^{-4}$ |
| `CS-AFL + Distill` (`seed=2026`) | `0.926 ± 0.013` | `0.9990` | `0.21%` | `98.85%` | `0.0082` | `0.0089` | `0.00%` | `0.00%` | `19.17%` | `27.50%` | $\chi^2=77.0, p<10^{-4}$ |
| **3-Seed Aggregate (`42/1337/2026`)** | **`0.934 ± 0.008`** | **`0.9991 ± 0.0001`** | **`0.25% ± 0.04%`** | **`98.97% ± 0.10%`** | **`0.0083 ± 0.0005`** | **`0.0086 ± 0.0003`** | **`0.46% ± 0.80%`** | **`0.00% ± 0.00%`** | **`21.14% ± 7.86%`** | **`28.08% ± 2.92%`** | All $p < 10^{-4}$ |
| **Prod: ONNX `int8_dynamic`** | **`0.926 ± 0.013`** | **`0.9988`** | **`0.29%`** | **`98.55%`** | **`0.0074`** | **`0.0113`** | **`1.39%`** | **`0.00%`** | `26.55%` | `28.50%` | $\chi^2=67.7, p<10^{-4}$ |
| Challenger: ONNX `int8_static_qdq` | `0.906 ± 0.015` | `0.9982` | `0.53%` | `97.86%` | `0.0089` | `0.0203` | `2.78%` | `0.00%` | `35.69%` | `30.75%` | $\chi^2=19.5, p<10^{-4}$ |

### 3.2 Four Scientific Takeaways from the Ablations

1. **Loss Ablation (`CE` vs `Focal` vs `CS-AFL`, controlled `seed=42`):** Standard `CE` achieves marginally higher 3-class argmax Macro-F1 (`0.940` vs `0.934`, well inside the `±0.013` bootstrap CI), **but** incurs **`+32%` higher in-domain FPR @ 95% TPR (`0.33%` vs `0.25%`)** and **`2×` higher OOD trigger-word false positives on `NotInject` (`29.20%` vs `14.45%`)**. Asymmetric FP weighting (`beta_fp = 2.25`, `gamma_0 = 2.5`) directly sharpens the low-FPR security operating regime (`F1_benign = 0.9912`).
2. **Distillation Ablation (`With` vs `No Distill`, controlled `seed=42`):** Removing Teacher logits slightly raises in-domain 3-class Macro-F1 (`0.942` vs `0.934`, within bootstrap CI), **but** degrades security calibration and OOD trigger robustness: `NotInject` FPR more than doubles (`14.45% → 30.97%`), in-domain `FPR @ 95% TPR` rises (`0.25% → 0.29%`), Brier score worsens (`0.0087 → 0.0095`), and logit temperature sharpens drastically ($T = 0.676 \rightarrow 0.448$). Meanwhile, the 44.1M distilled Student matches/exceeds its 86.0M Teacher in-domain (`0.934` vs `0.933` Macro-F1; `0.25%` vs `0.37%` FPR@95%TPR) at half the backbone size.
3. **Multi-Seed Stability (`42 / 1337 / 2026`):** In-domain ranking and security metrics are rock-solid across seeds: `ROC-AUC = 0.9991 ± 0.0001`, `FPR @ 95% TPR = 0.25% ± 0.04%`, `TPR @ 1% FPR = 98.97% ± 0.10%`, `XSTest Safe FPR = 0.00% ± 0.00%`. In contrast, `NotInject` FPR shows high threshold-induced variance (`21.14% ± 7.86%`) because Validation-tuned $\tau_{95}$ varies across seeds (`0.9036` on `seed1337` vs `0.9876` on `seed42`).
4. **Quantization Duel & Systems Physics:** On Zen 3 AVX2 (without AVX-512 VNNI), **`int8_dynamic`** strictly dominates **`int8_static_qdq`** in latency (`13.59 ms` vs `16.54 ms` P99 at `L=64`), fidelity (`98.76%` vs `95.92%` parity; `KL = 0.0064` vs `0.0573`), Macro-F1 (`0.926` vs `0.906`), and OOD `NotInject` FPR (`26.55%` vs `35.69%`). Live proxy tests confirm **`176.4 req/s`** peak throughput (`C = 16`), **`127.76 ms` P99 at `4,096` tokens** with 100% buried-attack blocking, **`326 MB`** non-root Docker image, and **`693–698 MB`** RSS driven by the unpruned `393.5 MB` 128k SentencePiece FP32 embedding table.

---

## 4. Threats to Validity

| Validity Dimension | Concrete Threat Identified | Empirical Mitigation & Residual Risk |
| :--- | :--- | :--- |
| **Internal Validity** | Near-duplicate leakage between splits/OOD or single-seed luck inflating performance. | Automated `pytest` zero-leakage gate (`602` MinHash LSH OOD purges, `StratifiedGroupKFold` over `23,712` clusters); 3 independent seeds (`42/1337/2026`) + McNemar tests ($p < 10^{-4}$). Residual threat: upstream label noise (~35% of Top-20 FPs/FNs in `PromptShield`/`deepset`/`hackaprompt`), intentionally left uncleaned in Test to preserve benchmark neutrality. |
| **External Validity** | Generalization to unseen obfuscations, non-English attacks, long RAG payloads, and enterprise technical traffic. | Verified zero-shot on 3 isolated OOD suites (`XSTest`, `NotInject`, `JBB`) + 270-vector OWASP red-team (`0.00%` ASR, `0.00%` UUID/SHA/JWT FPR) + buried mid-doc attacks up to `4,096` tokens. Residual threat: low non-English coverage (German roleplay FNs) and `26.55%` `NotInject` trigger-word FPR under INT8. |
| **Construct Validity** | Conflating **OWASP LLM01 (instruction hijacking)** with general **content toxicity/harm moderation**, or evaluating offline single-window (`L <= 256`) models as if they equal the online gateway. | Separated `XSTest`/`JBB` direct-harm behavior from LLM01 hijacking (explaining `28.5%` JBB TPR alongside `0.00%` XSTest Safe FPR) and measured both isolated ONNX (`13.59 ms` P99) and full live FastAPI proxy (`15.33 ms` short-benign P99, `127.76 ms` at `4,096` tok). |
| **Conclusion Validity** | Over-claiming small Macro-F1 differences across ablations or cherry-picking post-hoc test thresholds. | Reported 1,000-sample paired bootstrap 95% CIs (`±0.012–0.015`), 3-seed `mean ± std`, and locked all temperatures ($T$) and thresholds ($\tau_{95}, \tau_{1\%}$) strictly on Validation before touching Test or OOD. Explicitly noted where Macro-F1 deltas lie within overlapping CIs while security operating-point deltas (`FPR@95%TPR`, `NotInject`, `KL`) separate methods. |

---

## 5. High-Impact Jury Q&A (Anticipated Defense Questions)

1. **Q: Why did `CE` (`0.940`) and `No Distill` (`0.942`) get slightly higher Macro-F1 than `CS-AFL + Distill` (`0.934`) — did your custom loss and distillation fail?**  
   **A:** No — they succeeded on the metric that actually governs an inline security gateway. The `0.006–0.008` Macro-F1 delta lies strictly inside the `±0.013` 95% bootstrap CI (and is driven mainly by harmless `INJECTION ↔ JAILBREAK` confusion, both of which trigger `BLOCK`). On security-critical false-positive operating points, **CS-AFL + Distillation cuts `NotInject` trigger-word FPR in half** on `seed=42` (`14.45%` vs `29.20%` CE and `30.97%` No Distill), lowers in-domain `FPR @ 95% TPR` (`0.25%` vs `0.33%` CE / `0.29%` No Distill), improves Brier calibration (`0.0087` vs `0.0095`), and prevents logit over-sharpening ($T = 0.676$ vs $0.448$).

2. **Q: Your Milestone 1 spec targeted `< 500 MB` RSS, yet your live proxy measures `693–698 MB`. Why?**  
   **A:** Pure hardware-level memory physics that we measured and chose to report transparently rather than hide. `deberta-v3-small` has a `128,100`-token SentencePiece vocabulary whose FP32 embedding matrix alone weighs **`393.5 MB`** (`98.4M × 4 B`, `90.5%` of the `434.9 MB` ONNX file), whereas the 6-layer INT8 Transformer backbone is only **`~41 MB`**. Adding the ONNX Runtime CPU arena and FastAPI/Rust tokenizer brings RSS to `693 MB`, while the non-root Docker image itself is **`326 MB`**. In Milestone 3, domain vocabulary pruning (`128k → 32k`) cuts embeddings to `98 MB` and total RSS to `< 350 MB` without altering the backbone.

3. **Q: Why is `JailbreakBench` TPR only `28.5%` and `NotInject` FPR `26.55%` on the deployed INT8 model?**  
   **A:** Two distinct, empirically verified mechanisms. First, **construct boundary**: `JailbreakBench` contains mostly *unwrapped* direct harmful requests (*"Write a keylogger script"*) with zero prompt-injection or persona-hijacking wrapper — AegisLLM is an OWASP LLM01 firewall, not a general toxicity classifier (which is also why `XSTest Safe FPR = 0.00%`). Second, **quantization-induced OOD sensitivity**: dynamic INT8 preserves `98.76%` in-domain parity (`0.29%` FPR@95%TPR) but increases `NotInject` trigger-word FPR from `14.45%` (FP32) to `26.55%` (INT8) on sentences stuffed with uppercase override tokens (`POLICY`, `INSTRUCTIONS`). That directly motivates **Quantization-Aware Training (QAT)** + trigger-word hard negatives in Milestone 3.

4. **Q: Why did you choose Dynamic INT8 over Static QDQ INT8 when static quantization is often assumed faster?**  
   **A:** On x86 Zen 3 **AVX2** (no AVX-512 VNNI), QDQ inserts explicit `QuantizeLinear`/`DequantizeLinear` nodes around non-fused ops, causing Static QDQ to be both **slower** (`16.54 ms` vs **`13.59 ms`** P99 at `L=64`, failing the `15 ms` SLA) and **far less accurate** (`95.92%` vs **`98.76%`** agreement, `9×` higher KL divergence `0.0573` vs `0.0064`, and `0.53%` vs `0.29%` FPR@95%TPR) due to high-dynamic-range activations in DeBERTa-v3's disentangled relative attention.

5. **Q: Your red-team report mentions an initial `66.7%` false positive rate on UUIDs/SHA/JWTs. How did you fix it without creating a security bypass?**  
   **A:** Real-model red-teaming exposed that SentencePiece shatters high-entropy UUIDv4, MD5/SHA-256, and JWT strings into ~45 rare subwords that look statistically identical to adversarial GCG suffixes. In `Canonicalizer._prepare_neural_view`, we mask verified non-ASCII hex digests, UUIDs, and valid 3-part JWTs strictly inside the neural view (`hex checksum`, `uuid value`, `jwt token`) **while still base64url-decoding JWT payload claims into secondary inspection views** so JWT claim smuggling is caught. Result: **`0/15` (`0.00%`) identifier FPR** while keeping **`0.00%` ASR (`210/210` attacks blocked)**.
