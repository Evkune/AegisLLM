# CONTEXT_HANDOFF — AegisLLM (Milestone 2, SEDL 103001022)
## 1. Objectifs académiques & métier
- **Métier :** Proxy de sécurité in-line classifiant `BENIGN` / `PROMPT_INJECTION` / `JAILBREAK` (OWASP LLM01) sous SLA CPU `< 15 ms` P99 (0 GPU en prod).
- **Académique (ETSIINF / Prof. Sira Vegas) :** Valider le Milestone 2 avec rigueur empirique : falsification de baseline (TF-IDF + LinearSVC), zéro fuite entre splits, évaluation OOD (`XSTest`, `JailbreakBench`), calibration et significativité statistique (McNemar / bootstrap).
## 2. Décisions techniques figées
- **Modèle & GPU :** `microsoft/deberta-v3-small` (44M), entraîné sur Google Colab **Tesla T4** en **FP32 pure** (A100/L4 bloqués sur palier promo Jio ; FP16 provoque des `NaN` sur DeBERTa/Turing), `batch_size=16`, `gradient_accumulation_steps=2`, 4 époques (`configs/milestone2_deberta_v3_small.yaml`).
- **Données (`t4_optimized`) :** 36 000 prompts répartis sur 600 clusters invariants disjoints (`scripts/prepare_dataset.py`), split 70/15/15 (25 133 / 5 458 / 5 409) conservant le ratio 50% Benign / 30% Injection / 20% Jailbreak, 0 `group_id` partagé.
- **Optimisation & Export :** Loss CS-AFL (`CostSensitiveAsymmetricFocalLoss`, $\gamma_0=2.5, \beta_{FP}=2.25$), FGM sur `word_embeddings` ($\epsilon=0.50$), Model Soup (top-3 checkpoints), Temperature Scaling + export ONNX opset 18 INT8 dynamique.
## 3. Bugs connus & état actuel (`origin/main` @ `39dcf9d`)
- **Correctif non encore exécuté sur Colab :** Le `RuntimeError` CPU/GPU dans `src/aegis_dl/losses.py` est corrigé dans `39dcf9d` avec reload auto de `aegis_dl` ajouté en Cellule 6 de `notebooks/01_train_deberta_v3_small.ipynb`, mais reste à valider en exécution T4.
- **Distillation inactive en Cellule 6 :** `teacher_model` (`microsoft/deberta-v3-base`) est chargé en Cellule 5 mais non transmis à `HybridDistillationCSAFL` (qui bascule sur CS-AFL seul).
- **Environnement local & export :** `torch` absent de `.venv` local (11 tests passés, `tests/test_onnx_parity.py` skipped) ; export ONNX opset 18 de DeBERTa-v3 et test de parité (`rtol=1e-3, atol=1e-4`) en Cellule 9 non encore exécutés sur Colab.
## 4. Reste à faire
1. Sur Colab : exécuter `!git pull origin main` puis lancer la Cellule 6 (~18-22 min sur T4), suivie des Cellules 7 (Model Soup + calibration $T$), 8 (Test + figures 300 DPI) et 9 (Top-20 erreurs + export ONNX INT8).
2. Exécuter `scripts/evaluate_ood.py` sur les suites OOD (`Paul/XSTest`, `JailbreakBench`) et rapatrier les artefacts (`models/`, `reports/`).
3. Finaliser les livrables académiques dans `02_SEDL/milestones/` et mettre à jour `02_SEDL/defense_briefs/`.
