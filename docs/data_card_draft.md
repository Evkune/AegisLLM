# Data Card (Consolidated Draft) — AegisLLM Golden Evaluation Corpus

Following the data reporting framework proposed by Gebru et al. (2021) (*Datasheets for Datasets*).

## 1. Motivation & Provenance
- **Purpose:** Provide a leakage-free, highly curated dataset for fine-tuning and evaluating low-latency security classifiers on OWASP LLM01 threats, with built-in controls against false refusals on benign technical queries.
- **Academic Context:** Master of Science in Software Engineering, ETSIINF - Universidad Politécnica de Madrid (UPM) & INSA Rennes.
- **Constituent Repositories on Hugging Face:**
  - `neuralchemy/Prompt-injection-dataset` (Apache 2.0) — Core zero-leakage spine with `group_id` hashing and 3,500 DevOps hard negatives.
  - `hendzh/PromptShield` (Apache 2.0) — Delimiter-structured full-context injections.
  - `perplexity-ai/browsesafe-bench` (MIT) — Indirect HTML, CSS, and web DOM injection attacks.
  - `xTRam1/safe-guard-prompt-injection` & `jackhhao/jailbreak-classification` — Persona-adoption jailbreaks vs benign creative roleplay.
  - **Strictly Held-Out OOD Test Suites (Zero training overlap):** `Paul/XSTest` (NAACL 2024, CC-BY-4.0), `leolee99/NotInject` (MIT, 2025), and `JailbreakBench/JBB-Behaviors` (NeurIPS 2024, MIT).

## 2. Composition & Class Distribution

The compiled corpus comprises **24,000 in-domain instances** and **1,089 isolated OOD evaluation instances**:

| Stratum / Class | Volume ($N=24,000$) | Ratio | Composition & Role |
| :--- | :---: | :---: | :--- |
| **Class 0: BENIGN** | **12,000** | **50.0%** | 8,400 conversational queries + **3,600 DevOps Hard Negatives** (`kill -9`, `bypass cache`, `override method`) to suppress keyword bias. |
| **Class 1: PROMPT_INJECTION** | **7,200** | **30.0%** | 4,200 direct overrides & delimiter escapes + 3,000 indirect HTML/RAG injections. |
| **Class 2: JAILBREAK** | **4,800** | **20.0%** | 2,800 persona-hijacking (DAN/developer mode) + 2,000 obfuscated (Base64/ROT13) payloads. |
| **OOD Evaluation Suite** | **1,089** | — | `XSTest` (450 safe/unsafe pairs) + `NotInject` (339 trigger-word safe) + `JBB-Behaviors` (200 behaviors + 300 judge pairs). |

## 3. Preprocessing, Canonicalization & Hygiene Pipeline
1. **Zero-Width Character Extraction:** Stripping `[\u200B-\u200D\u2060\uFEFF\u00AD\u180E\uFFF9-\uFFFB]` and all Unicode Format (Cf) code points while recording anomaly counters.
2. **Unicode NFKC Normalization:** Canonical decomposition and composition mapping Cyrillic, Greek, and full-width homoglyphs into standard Latin characters.
3. **Adversarial Character-Spacing Collapse:** Detecting and collapsing spaced word evasion patterns (`I g n o r e   a l l   p r e v i o u s`) to counter the Cisco PromptGuard bypass vulnerability.
4. **Group-Aware Deduplication:** MinHash LSH ($0.85$ Jaccard similarity) ensuring no base prompt and its paraphrased or encoded variants cross between Train, Validation, and Test partitions.
