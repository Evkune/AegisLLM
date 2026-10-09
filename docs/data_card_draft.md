# Data Card (Preliminary Draft) — AegisLLM Evaluation Corpora

Following the data documentation framework proposed by Gebru et al. (2021) (*Datasheets for Datasets*).

## 1. Dataset Motivation and Scope
The evaluation datasets for AegisLLM are designed to benchmark adversarial injection detectors across diverse threat categories while rigorously tracking false refusal rates on benign, ambiguous queries.

## 2. Benchmark Composition

### A. TensorTrust (ICLR 2024)
- **Source:** Human-generated adversarial attacks captured during real-world competitive game sessions.
- **Role:** High-entropy realistic injection attempts bypassing basic prompt defenses.
- **Label:** `direct_injection` / `malicious`.

### B. JailbreakBench (NeurIPS 2024)
- **Source:** Curated safety benchmark containing standardized adversarial jailbreak prompts aimed at forcing LLMs to violate safety policies.
- **Role:** Evaluation of persona hijacking, developer-mode triggers, and systemic rule overrides.
- **Label:** `jailbreak` / `malicious`.

### C. XSTest (NAACL 2024)
- **Source:** Benchmark of safe conversational prompts that share superficial structural similarities with unsafe requests (e.g., historical inquiries, fictional scenarios).
- **Role:** Crucial control dataset to measure and minimize the False Positive Rate (FPR).
- **Label:** `benign` / `safe`.

## 3. Preprocessing & Partitioning
- **De-duplication:** Exact and near-duplicate removal across evaluation splits.
- **Normalization:** Unicode NFKC normalization and invisible zero-width character extraction.
- **Data Splitting:** Stratified chronological splits to prevent data leakage between training and evaluation phases.
