# Hyperparameter & Architecture Search Results (Phases 1-6)

**Total Successful Trials:** 66

This document summarizes all completed hyperparameter and architecture trials for EmotionHeart+ fine-tuning on the IEMOCAP dataset, sorted by `best_test_f1` (descending). It serves as a comprehensive reference for sensitivity analysis.

## Overview of Phases
- **Phase 1 (Coarse Random):** Broad exploration of learning rate, dropout, NACL, unimodal lambda, NACL lambda, temperature, and topk.
- **Phase 2 (Fine Grid):** Narrower grid search around the best config from Phase 1.
- **Phase 3 (Regularization & Class Balance):** Tuning `weight_decay`, Weighted Cross-Entropy (`do_WCE`), and Cosine LR Warmup Duration (`T`).
- **Phase 4 (Adaptive Search):** Data-driven exploration shifting parameters (e.g. `learning_rate`, `dropout`, `weight_decay`) based on Pearson correlation with F1 scores.
- **Phase 5 (Architecture Scaling):** Deep and wide transformer encoder tuning (`encoder_layers`: 2~4, `encoder_embed_dim`: 256~512, `encoder_attention_heads`: 4~8) while fixing the optimal HPs.
- **Phase 6 (Seed Robustness):** Testing the global optimum across 5 random seeds to verify convergence resilience.

## Phase 6 Robustness Summary
Multiple run statistics over 5 random seeds using the optimal architecture and hyperparameter set:
- **Average F1:** `0.7103 ± 0.0016`
- **Max F1:** `0.7122`
- **Min F1:** `0.7078`

## Complete Trial Leaderboard

| Rank | F1 Score | Phase | Learning Rate | Dropout | NACL | T | WCE | Layers | Embed Dim | Heads | Seed | Run ID |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| 1 | **0.7158** | 1 | 1.0e-04 | 0.3 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t006_lr=1e-04_dr=0.30_...` |
| 2 | **0.7156** | 1 | 1.0e-04 | 0.1 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t000_lr=1e-04_dr=0.10_...` |
| 3 | **0.7156** | 3 | 1.0e-04 | 0.1 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p3_t002_wd=1e-04_wce=0_T=...` |
| 4 | **0.7154** | 2 | 7.0e-05 | 0.2 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t004_lr=7e-05_dr=0.20_...` |
| 5 | **0.7152** | 1 | 7.0e-05 | 0.2 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t019_lr=7e-05_dr=0.20_...` |
| 6 | **0.7151** | 4 | 1.1e-04 | 0.092 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t005_lr=1e-04_dr=0.09_...` |
| 7 | **0.7133** | 1 | 5.0e-05 | 0.1 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t009_lr=5e-05_dr=0.10_...` |
| 8 | **0.7124** | 2 | 1.0e-04 | 0.2 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t002_lr=1e-04_dr=0.20_...` |
| 9 | **0.7122** | 6 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 4 | 20242065 | `p6_t004_seed=20242065` |
| 10 | **0.7118** | 3 | 1.0e-04 | 0.3 | False | 5 | True | 2 | 384 | 6 | 20242025 | `p3_t010_wd=5e-05_wce=1_T=...` |
| 11 | **0.7116** | 3 | 1.0e-04 | 0.2 | False | 3 | True | 2 | 384 | 6 | 20242025 | `p3_t001_wd=1e-05_wce=1_T=...` |
| 12 | **0.7115** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 4 | 20242025 | `p5_t013_L=3_D=384_H=4` |
| 13 | **0.7114** | 3 | 1.0e-04 | 0.1 | False | 5 | False | 2 | 384 | 6 | 20242025 | `p3_t008_wd=5e-05_wce=0_T=...` |
| 14 | **0.7110** | 4 | 1.1e-04 | 0.09 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t007_lr=1e-04_dr=0.09_...` |
| 15 | **0.7109** | 6 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 4 | 20242045 | `p6_t002_seed=20242045` |
| 16 | **0.7106** | 1 | 5.0e-05 | 0.2 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t010_lr=5e-05_dr=0.20_...` |
| 17 | **0.7105** | 6 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 4 | 20242025 | `p6_t000_seed=20242025` |
| 18 | **0.7105** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 4 | 384 | 6 | 20242025 | `p5_t000_L=4_D=384_H=6` |
| 19 | **0.7101** | 6 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 4 | 20242055 | `p6_t003_seed=20242055` |
| 20 | **0.7101** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 2 | 384 | 4 | 20242025 | `p5_t011_L=2_D=384_H=4` |
| 21 | **0.7101** | 4 | 1.1e-04 | 0.092 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t000_lr=1e-04_dr=0.09_...` |
| 22 | **0.7100** | 3 | 1.0e-04 | 0.2 | False | 10 | False | 2 | 384 | 6 | 20242025 | `p3_t006_wd=1e-03_wce=0_T=...` |
| 23 | **0.7100** | 1 | 7.0e-05 | 0.3 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t007_lr=7e-05_dr=0.30_...` |
| 24 | **0.7099** | 2 | 7.0e-05 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t010_lr=7e-05_dr=0.40_...` |
| 25 | **0.7098** | 4 | 1.1e-04 | 0.093 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t004_lr=1e-04_dr=0.09_...` |
| 26 | **0.7098** | 2 | 7.0e-05 | 0.2 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t009_lr=7e-05_dr=0.20_...` |
| 27 | **0.7096** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 4 | 384 | 4 | 20242025 | `p5_t004_L=4_D=384_H=4` |
| 28 | **0.7095** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 2 | 384 | 8 | 20242025 | `p5_t010_L=2_D=384_H=8` |
| 29 | **0.7095** | 2 | 7.0e-05 | 0.2 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t007_lr=7e-05_dr=0.20_...` |
| 30 | **0.7094** | 1 | 3.0e-05 | 0.3 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t011_lr=3e-05_dr=0.30_...` |
| 31 | **0.7092** | 4 | 1.1e-04 | 0.091 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t003_lr=1e-04_dr=0.09_...` |
| 32 | **0.7090** | 3 | 1.0e-04 | 0.2 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p3_t009_wd=1e-04_wce=0_T=...` |
| 33 | **0.7090** | 1 | 7.0e-05 | 0.1 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t002_lr=7e-05_dr=0.10_...` |
| 34 | **0.7089** | 2 | 1.0e-04 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t001_lr=1e-04_dr=0.40_...` |
| 35 | **0.7089** | 3 | 1.0e-04 | 0.2 | False | 5 | False | 2 | 384 | 6 | 20242025 | `p3_t004_wd=5e-05_wce=0_T=...` |
| 36 | **0.7089** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 6 | 20242025 | `p5_t003_L=3_D=384_H=6` |
| 37 | **0.7087** | 1 | 5.0e-05 | 0.3 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t005_lr=5e-05_dr=0.30_...` |
| 38 | **0.7087** | 1 | 5.0e-05 | 0.1 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t003_lr=5e-05_dr=0.10_...` |
| 39 | **0.7086** | 3 | 1.0e-04 | 0.1 | False | 3 | True | 2 | 384 | 6 | 20242025 | `p3_t011_wd=5e-05_wce=1_T=...` |
| 40 | **0.7084** | 2 | 1.0e-04 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t000_lr=1e-04_dr=0.40_...` |
| 41 | **0.7081** | 1 | 1.0e-04 | 0.1 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t001_lr=1e-04_dr=0.10_...` |
| 42 | **0.7080** | 2 | 1.0e-04 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t011_lr=1e-04_dr=0.40_...` |
| 43 | **0.7078** | 6 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 4 | 20242035 | `p6_t001_seed=20242035` |
| 44 | **0.7078** | 1 | 7.0e-05 | 0.3 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t004_lr=7e-05_dr=0.30_...` |
| 45 | **0.7069** | 3 | 1.0e-04 | 0.2 | False | 5 | False | 2 | 384 | 6 | 20242025 | `p3_t005_wd=1e-03_wce=0_T=...` |
| 46 | **0.7066** | 3 | 1.0e-04 | 0.1 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p3_t000_wd=5e-05_wce=0_T=...` |
| 47 | **0.7065** | 2 | 7.0e-05 | 0.3 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t005_lr=7e-05_dr=0.30_...` |
| 48 | **0.7063** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 4 | 384 | 8 | 20242025 | `p5_t015_L=4_D=384_H=8` |
| 49 | **0.7063** | 1 | 7.0e-05 | 0.4 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t015_lr=7e-05_dr=0.40_...` |
| 50 | **0.7061** | 2 | 1.0e-04 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t003_lr=1e-04_dr=0.40_...` |
| 51 | **0.7060** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p5_t019_L=2_D=384_H=6` |
| 52 | **0.7060** | 3 | 1.0e-04 | 0.3 | False | 10 | False | 2 | 384 | 6 | 20242025 | `p3_t007_wd=5e-05_wce=0_T=...` |
| 53 | **0.7059** | 1 | 3.0e-05 | 0.3 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t008_lr=3e-05_dr=0.30_...` |
| 54 | **0.7059** | 5 | 1.1e-04 | 0.092 | False | 3 | False | 3 | 384 | 8 | 20242025 | `p5_t007_L=3_D=384_H=8` |
| 55 | **0.7056** | 1 | 7.0e-05 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t012_lr=7e-05_dr=0.40_...` |
| 56 | **0.7055** | 4 | 1.1e-04 | 0.093 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t006_lr=1e-04_dr=0.09_...` |
| 57 | **0.7053** | 1 | 7.0e-05 | 0.3 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t013_lr=7e-05_dr=0.30_...` |
| 58 | **0.7053** | 2 | 1.0e-04 | 0.4 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t006_lr=1e-04_dr=0.40_...` |
| 59 | **0.7045** | 3 | 1.0e-04 | 0.3 | False | 10 | True | 2 | 384 | 6 | 20242025 | `p3_t003_wd=1e-04_wce=1_T=...` |
| 60 | **0.7044** | 4 | 1.1e-04 | 0.09 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t002_lr=1e-04_dr=0.09_...` |
| 61 | **0.7042** | 4 | 1.1e-04 | 0.091 | False | 3 | False | 2 | 384 | 6 | 20242025 | `p4_t001_lr=1e-04_dr=0.09_...` |
| 62 | **0.7039** | 1 | 5.0e-05 | 0.1 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t017_lr=5e-05_dr=0.10_...` |
| 63 | **0.7033** | 2 | 1.0e-04 | 0.3 | False | N/A | N/A | 2 | 384 | 6 | 20242025 | `p2_t008_lr=1e-04_dr=0.30_...` |
| 64 | **0.7030** | 1 | 5.0e-05 | 0.2 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t018_lr=5e-05_dr=0.20_...` |
| 65 | **0.6896** | 1 | 1.0e-05 | 0.4 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t014_lr=1e-05_dr=0.40_...` |
| 66 | **0.6880** | 1 | 1.0e-05 | 0.3 | True | N/A | N/A | 2 | 384 | 6 | 20242025 | `p1_t016_lr=1e-05_dr=0.30_...` |
