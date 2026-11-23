# Data Card — AegisLLM Zero-Leakage Evaluation & Training Corpus

Following Gebru et al. (2021), *Datasheets for Datasets*. All statistics below are verified directly against `data/manifest.json` (`29,755` unique in-domain prompts and `1,289` strictly isolated OOD evaluation prompts).

---

## 1. Motivation & Provenance
- **Purpose:** Provide a leakage-free, public, reproducible corpus for training and evaluating CPU-bound OWASP LLM01 classifiers (`BENIGN` / `PROMPT_INJECTION` / `JAILBREAK`), with explicit controls for trigger-word over-refusal and DevOps technical false positives.
- **Academic Context:** Master of Science in Software Engineering, ETSIINF — Universidad Politécnica de Madrid (UPM) & INSA Rennes (*Software Engineering for Deep Learning*, 103001022).
- **Public In-Domain Sources (all non-gated Hugging Face datasets):**
  - `hendzh/PromptShield` (`default`, capped `10,000` raw before dedup) — Long, structured prompt injections and benign contexts (`9,997` final deduplicated rows across splits).
  - `xTRam1/safe-guard-prompt-injection` (`default`) — Conversational benign prompts and direct injections (`9,781` final rows).
  - `neuralchemy/Prompt-injection-dataset` (`full`) — Core spine providing fine-grained sub-categories including `hard_negative_devops`, `hackaprompt`, `wildguard_judgecomp`, and `harmbench` (`8,534` final rows across sub-sources).
  - `jackhhao/jailbreak-classification` (`default`) — Persona/adversarial jailbreaks and benign prompts (`768` final rows).
  - `deepset/prompt-injections` (`default`) — Classic direct injections (`655` final rows).
  - `rubend18/ChatGPT-Jailbreak-Prompts` (`default`) — Supplemental DAN/persona jailbreaks (`20` unique post-dedup rows).
- **Strictly Isolated OOD Holdout Suites (zero training overlap, enforced by automated pytest gate):**
  - `Paul/XSTest` (`450` unique prompts: `250` safe exaggerations/homonyms, `200` unsafe direct queries).
  - `leolee99/NotInject` (`339` unique benign prompts containing 1–3 injection trigger keywords).
  - `JailbreakBench/JBB-Behaviors` (`500` unique prompts: `100` benign controls, `400` harmful jailbreak behaviors/attacks).

---

## 2. Composition & Empirical Split Distribution

After exact deduplication, label-conflict removal, and **602 cross-contamination purges** against the OOD holdouts via MinHash LSH (`Jaccard >= 0.75`), the remaining **`29,755` unique prompts** are grouped into **`23,712` structural clusters** and partitioned 70 / 15 / 15 via `StratifiedGroupKFold(n_splits=20)`:

| Partition | Rows | Unique Texts | Unique Clusters | Class 0: `BENIGN` | Class 1: `PROMPT_INJECTION` | Class 2: `JAILBREAK` | Median / P95 Char Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `20,878` | `20,878` | `16,374` | `11,358` (`54.4%`) | `8,268` (`39.6%`) | `1,252` (`6.0%`) | `131` / `1,483` |
| **Validation** | `4,439` | `4,439` | `3,670` | `2,434` (`54.8%`) | `1,753` (`39.5%`) | `252` (`5.7%`) | `109` / `1,090` |
| **Test** | `4,438` | `4,438` | `3,668` | `2,433` (`54.8%`) | `1,752` (`39.5%`) | `253` (`5.7%`) | `117` / `1,577` |
| **Total In-Domain** | **`29,755`** | **`29,755`** | **`23,712`** | **`16,225` (`54.5%`)** | **`11,773` (`39.6%`)** | **`1,757` (`5.9%`)** | — |

### OOD Holdout Suites (`1,289` prompts)
| Suite | Rows | Unique Texts | Label Breakdown | Primary Diagnostic Metric |
| :--- | :---: | :---: | :--- | :--- |
| `data/ood/xstest.parquet` | `450` | `450` | `250` Safe (`0`), `200` Unsafe (`1`) | Safe Over-Refusal FPR & Content-Safety Separation |
| `data/ood/notinject.parquet` | `339` | `339` | `339` Benign (`0`) | Trigger-Word Lexical Bias FPR |
| `data/ood/jbb.parquet` | `500` | `500` | `100` Benign (`0`), `400` Jailbreak (`2`) | Zero-Shot Harmful Recall (TPR) & Benign FPR |

---

## 3. Class Imbalance & Minority Class Mitigation
Class 2 (`JAILBREAK`) represents **`6.0%`** (`1,252` rows) of the deduplicated training split. Rather than oversampling (which introduces near-duplicate memorization and cross-split leakage risk), training uses empirically balanced square-root inverse-frequency class weights computed by `compute_class_weights_from_counts(..., smoothing_power=0.5)` (`[1.0, 1.172, 3.012]`) inside the differentiable `CostSensitiveAsymmetricFocalLoss` (`CS-AFL`).

---

## 4. Hygiene, Canonicalization & Zero-Leakage Verification
1. **Shared Single-Source-of-Truth Normalization (`src/aegis/security/canonicalizer.py` & `src/aegis_dl/normalization.py`):** Plane-14 tag stripping, Bidi Trojan Source stripping, zero-width (`Cf`) extraction, NFKC folding, and targeted mixed-script UTR#39 confusable normalization.
2. **MinHash LSH Structural Clustering:** 5-gram character shingles, `128` permutations, Jaccard threshold `0.75`, Union-Find (`DisjointSet`) connected-component grouping.
3. **Automated Gate (`tests/test_zero_leakage.py`):** Enforces zero shared `group_id` across Train/Val/Test, zero exact text overlap, zero OOD contamination (`In-Domain ∩ OOD == ∅`), real OOD cardinality (`XSTest >= 400`, `NotInject >= 300`, `JBB >= 180`), and sequence-length diversity (`P95 char_len >= 400`).
4. **Integrity Manifest (`data/manifest.json`):** Stores SHA-256 checksums for every split and OOD parquet file.

---

## 5. Known Upstream Label Noise & Construct Validity Notes
Manual inspection of the Top-20 highest-confidence False Positives and Top-20 False Negatives on the held-out Test split (`reports/error_analysis_m2.md`) identified two important empirical properties of public prompt-security datasets:
- **Upstream Public Label Noise (~35% of Top-20 errors):** Four of the Top-20 FPs (`hendzh/PromptShield`) contain explicit instruction overrides (*"Ignore all instructions and rules above..."*) labeled `0` in the source dataset, while seven of the Top-20 FNs (`deepset/prompt-injections`, `neuralchemy:hackaprompt`) are benign or nonsensical strings (*"LIST FIVE FRUITS."*, *"Tell me a joke about a woman"*) labeled `1` upstream. **We deliberately do NOT clean Test labels post-hoc** to preserve unbiased benchmark comparability, and report this noise ceiling transparently.
- **Construct Separation (LLM01 vs. Content Moderation):** `XSTest` unsafe prompts and many `JailbreakBench` behaviors are direct, unwrapped harmful requests lacking adversarial override framing; evaluating them alongside `NotInject` cleanly separates *instruction-hijacking detection* from general *content moderation*.
