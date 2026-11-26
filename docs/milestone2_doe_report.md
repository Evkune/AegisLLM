# AegisLLM — Milestone 2: Design of Experiments (DOE) Report

**Course:** Software Engineering for Deep Learning (103001022) — ETSIINF, Universidad Politécnica de Madrid  
**Authors:** Evan Gendrot & Amaury Dubuc  
**System Under Test:** AegisLLM (`microsoft/deberta-v3-small`, 44.1M backbone — OWASP LLM01 Prompt Injection & Jailbreak Firewall)  
**Statistical Dataset:** `reports/doe/doe_raw_data.csv` | **Analysis Script:** `scripts/analyze_doe_anova.py`

---

## 1.- Experiment Planning and Definition

### 1.1 Experimental Goal (GQM Framing)
**Analyze** the training loss formulation and teacher-to-student knowledge distillation **for the purpose of** evaluating their main and interaction effects **with respect to** low-FPR in-domain security detection, out-of-distribution (OOD) benign trigger-word false positives, and 3-class classification balance **from the point of view of** an AI security engineer **in the context of** an inline CPU-only `DeBERTa-v3-Small` prompt-injection gateway (`L <= 256`).

### 1.2 Response Variables (Dependent Variables)
Three continuous response variables in $[0, 1]$ are evaluated for every experimental run after locking temperature calibration ($T$) and operating threshold ($\tau_{95}$) strictly on the isolated Validation split ($N = 4{,}439$):

| Role | Variable Name (`CSV` column) | Operational Definition | Dataset Evaluated | Optimization Direction |
| :--- | :--- | :--- | :--- | :---: |
| **Primary Security** | `fpr_at_95_tpr` | False Positive Rate on benign prompts when evaluated at the Validation-locked 95% attack recall operating point $\tau_{95}$ (`s_mal = 1 - P(BENIGN) >= tau_95`). | Held-out Test ($N = 4{,}438$; $2{,}433$ benign) | **Minimize** |
| **Co-Primary OOD** | `notinject_fpr` | False Positive Rate on benign prompts intentionally enriched with 1–3 injection trigger keywords (e.g., `ignore`, `system`, `policy`) at the same locked $\tau_{95}$. | Isolated OOD `NotInject` ($N = 339$; 100% benign) | **Minimize** |
| **Utility / Balance** | `macro_f1` | Unweighted 3-class Macro-F1 across `BENIGN` (0), `PROMPT_INJECTION` (1), and `JAILBREAK` (2) under `argmax` over calibrated probabilities. | Held-out Test ($N = 4{,}438$) | **Maximize** |
| *Diagnostic (raw CSV)* | `roc_auc`, `ece` | Binary malicious discrimination (`ROC-AUC`) and 15-bin Expected Calibration Error (`ECE`) after post-hoc Temperature Scaling. | Held-out Test ($N = 4{,}438$) | Max / Min |

### 1.3 Factors and Levels (Treatments)
Two controlled categorical factors are varied ($\le 3$ factors as required), yielding $2 \times 2 = 4$ treatment combinations:

| Factor | Symbol | Levels (Treatments) | Technical Specification |
| :--- | :---: | :--- | :--- |
| **Factor A — Loss Function** | `loss` | `ce` (baseline)<br>`cs_afl` (proposed) | **`ce`:** Standard unweighted multi-class Cross-Entropy.<br>**`cs_afl`:** Cost-Sensitive Asymmetric Focal Loss with class weights $[1.0, 1.35, 3.01]$, focusing exponents $\gamma = [2.5, 1.5, 2.0]$, asymmetric FP penalty $\beta_{\text{fp}} = 2.25$, and label smoothing $\epsilon = 0.03$. |
| **Factor B — Knowledge Distillation** | `distillation` | `off` (baseline)<br>`on` (proposed) | **`off`:** Supervised training from hard ground-truth labels only ($\alpha_{\text{distill}} = 0.0$).<br>**`on`:** Hybrid KL-divergence soft-target distillation ($\alpha_{\text{distill}} = 0.40, T_{\text{kd}} = 2.0$) from the frozen fine-tuned `microsoft/deberta-v3-base` teacher (`models/teacher_train_logits.pt`). |

### 1.4 Blocking Variables
**No blocking variable is introduced** in the primary design (Completely Randomized Factorial Design — CRFD):
- All 20 experimental runs execute on identical hardware and numerical precision (single Google Colab GPU SKU, deterministic **FP32**) over the exact same frozen zero-leakage train/val/test/OOD partitions and the exact same cached Teacher logits file.
- The 5 random seeds `{42, 1337, 2026, 777, 999}` act as **independent stochastic replications** per treatment cell (governing classifier head initialization, dropout masks, and mini-batch length-bucket shuffling) rather than heterogeneous subjects with shared intrinsic characteristics across distinct optimization landscapes.
- *Design Safeguard:* Hardware GPU SKU would act as a nuisance blocking factor if mixed (e.g., T4 vs. L4); this nuisance is eliminated by holding GPU SKU and FP32 precision strictly constant across all 20 runs.

### 1.5 Contextual Variables Held Constant
Every factor outside `loss` and `distillation` is held strictly constant across all 20 DOE runs (`reports/predictions_doe_fast/`). Once the factorial DOE identifies the statistically optimal `(loss, distillation)` combination, that winning configuration feeds into the full Milestone 2 production hardening recipe (3 epochs + FGM adversarial training + Greedy Model Soup + ONNX INT8 Dynamic export).

| Contextual Variable | Fixed Value / Specification (DOE Protocol) |
| :--- | :--- |
| Student Backbone | `microsoft/deberta-v3-small` (`44.1M` params, 6 Transformer layers, 128k SentencePiece vocab) |
| Sequence Windowing | `max_sequence_length = 256`, `pad_to_multiple_of = 8`, `truncation = True` |
| Dataset & Zero-Leakage Split | `data/processed/{train,val,test}.parquet` (`20,878` / `4,439` / `4,438`); `602` MinHash LSH OOD duplicates purged (`Jaccard >= 0.75`); `StratifiedGroupKFold(n_splits=20)` over `23,712` clusters |
| OOD Stress Suites | `data/ood/notinject.parquet` ($N=339$), `xstest.parquet` ($N=450$), `jbb.parquet` ($N=500$) |
| Training Schedule | `2` epochs (`1,304` optimizer steps), cosine decay with `10%` linear warmup, uniform end-of-epoch validation checkpoint selection (`keep_ckpts = False`) |
| Batching & Effective Batch | `LengthBucketedBatchSampler(batch_size=16, mega_mult=40)` + `grad_accum = 2` (effective batch = `32`) |
| Optimizer & LLRD | `AdamW` (`backbone_lr = 2.5e-5`, `head_lr = 8.0e-5`, `llrd_decay = 0.90`, `betas = (0.9, 0.98)`, `eps = 1e-6`, `weight_decay = 0.01`, `max_grad_norm = 1.0`) |
| Adversarial Training (FGM) | Held constant at `OFF` (`--disable-fgm`) during the factorial DOE to isolate `Loss × Distillation` without second-order adversarial perturbation confounding; enabled in final production hardening |
| Teacher Model (for `distillation=on`) | Frozen `microsoft/deberta-v3-base` (86.0M) soft logits cached in `models/teacher_train_logits.pt` (`float16`, shape `[20878, 3]`) |
| Calibration & Operating Point | Post-hoc `TemperatureScaler` fitted strictly on Validation (`val.parquet`); $\tau_{95}$ and $\tau_{1\%}$ locked on Validation ROC before blind Test/OOD inference |
| Hardware & Numerical Precision | Single uniform Google Colab GPU SKU across all 20 runs, explicit **FP32** (`model.float()`) |

### 1.6 Hypotheses Formulation ($\alpha = 0.05$)
For each response variable $Y \in \{\text{fpr\_at\_95\_tpr},\ \text{notinject\_fpr},\ \text{macro\_f1}\}$:
- **Main Effect of Factor A (Loss):**
  - $H_{0,A}: \mu_{\text{ce}} = \mu_{\text{cs\_afl}}$ (loss function has no effect on mean $Y$)
  - $H_{1,A}: \mu_{\text{ce}} \neq \mu_{\text{cs\_afl}}$
- **Main Effect of Factor B (Distillation):**
  - $H_{0,B}: \mu_{\text{off}} = \mu_{\text{on}}$ (knowledge distillation has no effect on mean $Y$)
  - $H_{1,B}: \mu_{\text{off}} \neq \mu_{\text{on}}$
- **Interaction Effect (Loss $\times$ Distillation):**
  - $H_{0,AB}: (\alpha\beta)_{ij} = 0\ \forall i, j$ (the effect of Loss does not depend on whether Distillation is enabled)
  - $H_{1,AB}: \exists\, i, j\text{ such that }(\alpha\beta)_{ij} \neq 0$

---

## 2.- Experimental Design

### 2.1 Design Type and Sample Size
- **Design Structure:** Balanced **$2 \times 2$ Full Factorial Design** without blocks (Completely Randomized Factorial Design).
- **Replications per Treatment Cell:** $r = 5$ independent stochastic replications (`seed ∈ {42, 1337, 2026, 777, 999}`), satisfying the minimum requirement of $\ge 5$ replications per factor-level combination.
- **Total Experimental Runs:** $N = a \times b \times r = 2 \times 2 \times 5 = \mathbf{20}$ runs.
- **Degrees of Freedom Allocation:** $df_{\text{Loss}} = 1$, $df_{\text{Distill}} = 1$, $df_{\text{Loss} \times \text{Distill}} = 1$, $df_{\text{Residual}} = N - ab = 16$, $df_{\text{Total}} = 19$.

| Treatment Cell | Factor A (`loss`) | Factor B (`distillation`) | Replications ($r = 5$ seeds) | Run ID Pattern |
| :---: | :---: | :---: | :--- | :--- |
| $T_{11}$ | `ce` | `off` | `42, 1337, 2026, 777, 999` | `deberta_v3_small_ce_no_distill_seed{s}` |
| $T_{12}$ | `ce` | `on` | `42, 1337, 2026, 777, 999` | `deberta_v3_small_ce_seed{s}` |
| $T_{21}$ | `cs_afl` | `off` | `42, 1337, 2026, 777, 999` | `deberta_v3_small_cs_afl_no_distill_seed{s}` |
| $T_{22}$ | `cs_afl` | `on` | `42, 1337, 2026, 777, 999` | `deberta_v3_small_cs_afl_seed{s}` |

### 2.2 Experimental Operation
1. **Pre-Experiment Data Hygiene Gate:** Freeze `data/processed/{train,val,test}.parquet` and `data/ood/*.parquet` and verify zero cluster/OOD leakage via `pytest tests/test_zero_leakage.py`.
2. **Teacher Logit Caching (Executed Once):** Load the verified `models/teacher_train_logits.pt` (`20,878 × 3`, `float16`) produced by Stage 1 fine-tuning of `microsoft/deberta-v3-base`, guaranteeing identical soft supervision for all 10 `distillation=on` runs.
3. **Randomized Resilient Batch Execution (`scripts/run_doe_batch.py`):**
   - Permute the 20-run matrix using a recorded PRNG seed (`--order-seed 2026`, persisted in `reports/doe/run_order.json`) to neutralize chronological hardware drift across Colab sessions.
   - For each run: call `seed_everything(seed)`, instantiate a fresh pretrained `microsoft/deberta-v3-small` model in FP32, train with the assigned `(loss, distillation)` treatment, select the best checkpoint strictly by Validation loss, fit `TemperatureScaler` on Validation logits, lock $\tau_{95}$ and $\tau_{1\%}$ on Validation calibrated probabilities, perform a single blind inference pass on Test and OOD suites, persist immutable `<run_key>_test.npz` and `<run_key>_meta.json` artifacts immediately (with Drive backup / Colab auto-download and a 4-minute kernel keep-alive heartbeat), and explicitly free GPU memory (`gc.collect()` + `torch.cuda.empty_cache()`).
4. **Automated Statistical Pipeline (`scripts/analyze_doe_anova.py`):** Load all 20 `.npz` / `_meta.json` artifacts, export the tidy raw dataset `reports/doe/doe_raw_data.csv`, compute descriptive statistics, render 300 DPI single-factor boxplots, Q-Q residual plots, and interaction plots, validate ANOVA assumptions (Shapiro-Wilk + Levene), compute Type II/III ANOVA tables and Tukey HSD post-hoc comparisons, and populate Section 3.

### 2.3 Control of Threats to Internal Validity

| Internal Validity Threat | Concrete Risk in Deep Learning DOE | Control Mechanism Applied |
| :--- | :--- | :--- |
| **History / Hardware Drift** | Thermal throttling or differing Colab container allocations over multi-hour GPU sessions could bias earlier vs. later treatments. | Execution order of all 20 runs is fully randomized (`--order-seed 2026`, logged in `reports/doe/run_order.json`), and GPU SKU + FP32 precision are held strictly uniform. |
| **Maturation / Carry-Over** | Weights, optimizer momentum states, or CUDA caches leaking between consecutive training runs inside one session. | Fresh `AutoModelForSequenceClassification.from_pretrained`, fresh `AdamW`/scheduler, fresh RNG seeding (`seed_everything`), and explicit `gc.collect()` + `torch.cuda.empty_cache()` after every run. |
| **Instrumentation** | Discrepancies in metric definitions or post-hoc threshold tuning across runs. | Single unified evaluation module (`aegis_dl.metrics.compute_comprehensive_metrics` + `scripts/analyze_doe_anova.py`); all thresholds ($\tau_{95}$) and temperatures ($T$) are locked strictly on Validation before Test/OOD scoring. |
| **Selection / Data Leakage** | Near-duplicate prompt templates crossing Train/Val/Test or OOD boundaries, artificially inflating metrics or compressing variance. | MinHash LSH cross-contamination purge (`602` OOD duplicates removed, Jaccard $\ge 0.75$) + `StratifiedGroupKFold(20)` over `23,712` clusters, enforced by automated `pytest` gates. |
| **Stochasticity / Lucky-Seed Bias** | Single-seed comparisons mistaking initialization/shuffle noise for a true treatment effect. | Balanced $r = 5$ replications per cell (`N = 20` runs, $df_{\text{error}} = 16$), enabling formal variance estimation ($MS_{\text{error}}$) and $F$-tests. |
| **Testing / Repeated Evaluation** | Repeatedly inspecting Test/OOD scores during hyperparameter tuning. | Held-out Test (`N = 4,438`) and OOD (`NotInject`, `N = 339`) are evaluated in a single blind pass only after training and Validation calibration terminate. |
| **Checkpoint-Selection Symmetry** | Asymmetric validation polling intervals across treatments could inflate one cell's best validation state. | All 20 DOE runs execute in a dedicated isolated suite (`reports/predictions_doe_fast/`) with identical `epochs=2`, `FGM=OFF`, and uniform end-of-epoch `keep_ckpts=False` selection — zero carry-over from Phase 1. |

---

## 3.- Analysis and Interpretation of Results

<!-- DOE_RESULTS_START -->

> **[PENDING COMPLETE $N = 20$ DOE RUNS]**  
> Once all 20 runs (`2 × 2 × 5`) are present in `reports/predictions/`, execute:
> ```bash
> .venv/bin/python scripts/analyze_doe_anova.py --update-report docs/milestone2_doe_report.md
> ```
> This block will be populated automatically with:
> - **3.1** Complete 20-row Raw Experimental Data Table (synchronized with `reports/doe/doe_raw_data.csv`)
> - **3.2** Descriptive Statistics (`Mean`, `SD`, `Median`, `IQR`, `Min`, `Max`) by Factor and by Cell + separate 300 DPI Factor Boxplots (`loss` and `distillation`) for `fpr_at_95_tpr`, `notinject_fpr`, and `macro_f1`
> - **3.3** Model Validation Table & 300 DPI Normal Probability (Q-Q) Plots of Residuals + Shapiro-Wilk & Levene's Test of Homogeneity of Variances
> - **3.4** Complete Two-Way ANOVA Tables (`Loss`, `Distillation`, `Loss:Distillation`, `Residual`, `Total` with `SS, df, MS, F, p-value, η², η_p²`), 300 DPI Interaction Plots, and formal statistical interpretations
> - **3.5** Statistically justified Recommendation of the Best Treatment Combination

<!-- DOE_RESULTS_END -->

---

### 3.6 Statistical Tool Options & Verification Screenshots

To guarantee full auditability against standard statistical software (JASP / Jamovi / SPSS), `reports/doe/doe_raw_data.csv` is formatted in tidy tabular layout (`run_id, loss, distillation, seed, macro_f1, roc_auc, fpr_at_95_tpr, notinject_fpr, ece`) and cross-verified with both Python (`statsmodels` OLS Type II/III ANOVA + `scipy.stats`) and interactive GUI statistical tooling.

#### Reproducible Python Statistical Specification
- **Command:** `.venv/bin/python scripts/analyze_doe_anova.py --preds-dir reports/predictions_doe_fast --update-report docs/milestone2_doe_report.md`
- **Model Formula:** `response ~ C(loss, Sum) * C(distillation, Sum)` (balanced $2 \times 2 \times 5$; Type II $\equiv$ Type III SS)
- **Diagnostic Tests:** Shapiro-Wilk normality test on ANOVA residuals (`scipy.stats.shapiro`), Normal Probability Q-Q plot (`scipy.stats.probplot`), Levene's test for equality of variances (`scipy.stats.levene(..., center="mean")` and Brown-Forsythe `center="median"`), and Tukey HSD post-hoc (`statsmodels.stats.multicomp.pairwise_tukeyhsd`).

#### GUI Statistical Tool Options Screenshots (JASP / Jamovi / SPSS)

> **[PLACEHOLDER — SCREENSHOT 1: ANOVA Model & Factor Options]**  
> *Insert screenshot showing: `ANOVA -> Classical ANOVA`, Dependent Variable (`fpr_at_95_tpr` / `notinject_fpr` / `macro_f1`), Fixed Factors (`loss`, `distillation`), Model terms (`loss`, `distillation`, `loss * distillation`), Sum of Squares (`Type III`), Display (`Descriptive statistics`, `Estimates of effect size: η², ω²`).*  
> `![JASP/Jamovi ANOVA Model Options](../reports/doe/screenshots/01_anova_model_options.png)`

> **[PLACEHOLDER — SCREENSHOT 2: Assumption Checks Options]**  
> *Insert screenshot showing: `Assumption Checks -> Homogeneity tests (Levene's)` and `Q-Q plot of residuals` checked.*  
> `![JASP/Jamovi Assumption Checks Options](../reports/doe/screenshots/02_assumption_checks_options.png)`

> **[PLACEHOLDER — SCREENSHOT 3: Descriptive Plots & Factor Boxplots Options]**  
> *Insert screenshot showing: `Descriptives / Descriptive Plots` configuration (Horizontal axis: `loss`, Separate lines: `distillation`, Error bars: `95% CI`) and individual factor boxplot settings.*  
> `![JASP/Jamovi Plots Options](../reports/doe/screenshots/03_plots_options.png)`
