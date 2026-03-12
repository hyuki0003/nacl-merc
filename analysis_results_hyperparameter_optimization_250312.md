# Hyperparameter Search Results (Phases 1-4)

**Total Successful Trials:** 52

This document summarizes all completed hyperparameter trials for EmotionHeart+ fine-tuning on the IEMOCAP dataset, sorted by `best_test_f1` (descending). It serves as a comprehensive reference for sensitivity analysis.

## Overview of Phases
- **Phase 1 (Coarse Random):** Broad exploration of learning rate, dropout, NACL, unimodal lambda, NACL lambda, temperature, and topk.
- **Phase 2 (Fine Grid):** Narrower grid search around the best config from Phase 1.
- **Phase 3 (Regularization & Class Balance):** Tuning `weight_decay`, Weighted Cross-Entropy (`do_WCE`), and Cosine LR Warmup Duration (`T`).
- **Phase 4 (Adaptive Search):** Data-driven exploration shifting parameters (e.g. `learning_rate`, `dropout`, `weight_decay`) based on Pearson correlation with F1 scores.

## Sorted Results

| Rank | F1 Score | Phase | Learning Rate | Dropout | Weight Decay | do_NACL | Unimodal λ | NACL λ | do_WCE | T | Run ID |
|:---:|:---:|:---:|:---|:---|:---|:---:|:---|:---|:---:|:---:|:---|
| 1 | **0.7158** | 1 | 1.0e-04 | 0.3 | N/A | False | 0.1 | 0.5 | N/A | N/A | `p1_t006_lr=1e-04_dr=0.30_nacl=0_ul=0.10_nl=0.5...` |
| 2 | **0.7156** | 1 | 1.0e-04 | 0.1 | N/A | False | 0.1 | 0.1 | N/A | N/A | `p1_t000_lr=1e-04_dr=0.10_nacl=0_ul=0.10_nl=0.1...` |
| 3 | **0.7156** | 3 | 1.0e-04 | 0.1 | 1.0e-04 | False | 0.1 | 0.5 | False | 3 | `p3_t002_wd=1e-04_wce=0_T=3_dr=0.10` |
| 4 | **0.7154** | 2 | 7.0e-05 | 0.2 | N/A | False | 0.1 | 0.3 | N/A | N/A | `p2_t004_lr=7e-05_dr=0.20_nacl=0_ul=0.10_nl=0.3...` |
| 5 | **0.7152** | 1 | 7.0e-05 | 0.2 | N/A | False | 0.1 | 0.5 | N/A | N/A | `p1_t019_lr=7e-05_dr=0.20_nacl=0_ul=0.10_nl=0.5...` |
| 6 | 0.7151 | 4 | 1.1e-04 | 0.09 | 9.9e-05 | False | 0.10 | 0.46 | False | 3 | `p4_t005_lr=1e-04_dr=0.09_wd=1e-04_ul=0.10` |
| 7 | 0.7133 | 1 | 5.0e-05 | 0.1 | N/A | True | 0.1 | 0.1 | N/A | N/A | `p1_t009_lr=5e-05_dr=0.10_nacl=1_ul=0.10_nl=0.1...` |
| 8 | 0.7124 | 2 | 1.0e-04 | 0.2 | N/A | False | 0.05 | 0.3 | N/A | N/A | `p2_t002_lr=1e-04_dr=0.20_nacl=0_ul=0.05_nl=0.3...` |
| 9 | 0.7118 | 3 | 1.0e-04 | 0.3 | 5.0e-05 | False | 0.1 | 0.5 | True | 5 | `p3_t010_wd=5e-05_wce=1_T=5_dr=0.30` |
| 10 | 0.7116 | 3 | 1.0e-04 | 0.2 | 1.0e-05 | False | 0.1 | 0.5 | True | 3 | `p3_t001_wd=1e-05_wce=1_T=3_dr=0.20` |
| 11 | 0.7114 | 3 | 1.0e-04 | 0.1 | 5.0e-05 | False | 0.1 | 0.5 | False | 5 | `p3_t008_wd=5e-05_wce=0_T=5_dr=0.10` |
| 12 | 0.7110 | 4 | 1.1e-04 | 0.09 | 1.1e-04 | False | 0.10 | 0.49 | False | 3 | `p4_t007_lr=1e-04_dr=0.09_wd=1e-04_ul=0.10` |
| 13 | 0.7106 | 1 | 5.0e-05 | 0.2 | N/A | False | 0.05 | 0.1 | N/A | N/A | `p1_t010_lr=5e-05_dr=0.20_nacl=0_ul=0.05_nl=0.1...` |
| 14 | 0.7101 | 4 | 1.1e-04 | 0.09 | 8.9e-05 | False | 0.10 | 0.49 | False | 3 | `p4_t000_lr=1e-04_dr=0.09_wd=9e-05_ul=0.10` |
| 15 | 0.7100 | 3 | 1.0e-04 | 0.2 | 1.0e-03 | False | 0.1 | 0.5 | False | 10 | `p3_t006_wd=1e-03_wce=0_T=10_dr=0.20` |
| 16 | 0.7100 | 1 | 7.0e-05 | 0.3 | N/A | True | 0.1 | 0.1 | N/A | N/A | `p1_t007_lr=7e-05_dr=0.30_nacl=1_ul=0.10_nl=0.1...` |
| 17 | 0.7099 | 2 | 7.0e-05 | 0.4 | N/A | False | 0.1 | 0.3 | N/A | N/A | `p2_t010_lr=7e-05_dr=0.40_nacl=0_ul=0.10_nl=0.3...` |
| 18 | 0.7098 | 4 | 1.1e-04 | 0.09 | 1.0e-04 | False | 0.10 | 0.54 | False | 3 | `p4_t004_lr=1e-04_dr=0.09_wd=1e-04_ul=0.10` |
| 19 | 0.7098 | 2 | 7.0e-05 | 0.2 | N/A | False | 0.1 | 0.5 | N/A | N/A | `p2_t009_lr=7e-05_dr=0.20_nacl=0_ul=0.10_nl=0.5...` |
| 20 | 0.7095 | 2 | 7.0e-05 | 0.2 | N/A | False | 0.3 | 0.5 | N/A | N/A | `p2_t007_lr=7e-05_dr=0.20_nacl=0_ul=0.30_nl=0.5...` |
| 21 | 0.7094 | 1 | 3.0e-05 | 0.3 | N/A | True | 0.1 | 0.5 | N/A | N/A | `p1_t011_lr=3e-05_dr=0.30_nacl=1_ul=0.10_nl=0.5...` |
| 22 | 0.7092 | 4 | 1.1e-04 | 0.09 | 1.1e-04 | False | 0.10 | 0.44 | False | 3 | `p4_t003_lr=1e-04_dr=0.09_wd=1e-04_ul=0.10` |
| 23 | 0.7090 | 3 | 1.0e-04 | 0.2 | 1.0e-04 | False | 0.1 | 0.5 | False | 3 | `p3_t009_wd=1e-04_wce=0_T=3_dr=0.20` |
| 24 | 0.7090 | 1 | 7.0e-05 | 0.1 | N/A | True | 0.5 | 0.1 | N/A | N/A | `p1_t002_lr=7e-05_dr=0.10_nacl=1_ul=0.50_nl=0.1...` |
| 25 | 0.7089 | 2 | 1.0e-04 | 0.4 | N/A | False | 0.3 | 0.5 | N/A | N/A | `p2_t001_lr=1e-04_dr=0.40_nacl=0_ul=0.30_nl=0.5...` |
| 26 | 0.7089 | 3 | 1.0e-04 | 0.2 | 5.0e-05 | False | 0.1 | 0.5 | False | 5 | `p3_t004_wd=5e-05_wce=0_T=5_dr=0.20` |
| 27 | 0.7087 | 1 | 5.0e-05 | 0.3 | N/A | True | 0.5 | 0.5 | N/A | N/A | `p1_t005_lr=5e-05_dr=0.30_nacl=1_ul=0.50_nl=0.5...` |
| 28 | 0.7087 | 1 | 5.0e-05 | 0.1 | N/A | True | 0.5 | 0.3 | N/A | N/A | `p1_t003_lr=5e-05_dr=0.10_nacl=1_ul=0.50_nl=0.3...` |
| 29 | 0.7086 | 3 | 1.0e-04 | 0.1 | 5.0e-05 | False | 0.1 | 0.5 | True | 3 | `p3_t011_wd=5e-05_wce=1_T=3_dr=0.10` |
| 30 | 0.7084 | 2 | 1.0e-04 | 0.4 | N/A | False | 0.3 | 0.5 | N/A | N/A | `p2_t000_lr=1e-04_dr=0.40_nacl=0_ul=0.30_nl=0.5...` |
| 31 | 0.7081 | 1 | 1.0e-04 | 0.1 | N/A | False | 0.05 | 0.1 | N/A | N/A | `p1_t001_lr=1e-04_dr=0.10_nacl=0_ul=0.05_nl=0.1...` |
| 32 | 0.7080 | 2 | 1.0e-04 | 0.4 | N/A | False | 0.1 | 0.3 | N/A | N/A | `p2_t011_lr=1e-04_dr=0.40_nacl=0_ul=0.10_nl=0.3...` |
| 33 | 0.7078 | 1 | 7.0e-05 | 0.3 | N/A | True | 0.05 | 0.3 | N/A | N/A | `p1_t004_lr=7e-05_dr=0.30_nacl=1_ul=0.05_nl=0.3...` |
| 34 | 0.7069 | 3 | 1.0e-04 | 0.2 | 1.0e-03 | False | 0.1 | 0.5 | False | 5 | `p3_t005_wd=1e-03_wce=0_T=5_dr=0.20` |
| 35 | 0.7066 | 3 | 1.0e-04 | 0.1 | 5.0e-05 | False | 0.1 | 0.5 | False | 3 | `p3_t000_wd=5e-05_wce=0_T=3_dr=0.10` |
| 36 | 0.7065 | 2 | 7.0e-05 | 0.3 | N/A | False | 0.1 | 0.5 | N/A | N/A | `p2_t005_lr=7e-05_dr=0.30_nacl=0_ul=0.10_nl=0.5...` |
| 37 | 0.7063 | 1 | 7.0e-05 | 0.4 | N/A | True | 0.5 | 0.3 | N/A | N/A | `p1_t015_lr=7e-05_dr=0.40_nacl=1_ul=0.50_nl=0.3...` |
| 38 | 0.7061 | 2 | 1.0e-04 | 0.4 | N/A | False | 0.05 | 0.3 | N/A | N/A | `p2_t003_lr=1e-04_dr=0.40_nacl=0_ul=0.05_nl=0.3...` |
| 39 | 0.7060 | 3 | 1.0e-04 | 0.3 | 5.0e-05 | False | 0.1 | 0.5 | False | 10 | `p3_t007_wd=5e-05_wce=0_T=10_dr=0.30` |
| 40 | 0.7059 | 1 | 3.0e-05 | 0.3 | N/A | True | 0.3 | 0.3 | N/A | N/A | `p1_t008_lr=3e-05_dr=0.30_nacl=1_ul=0.30_nl=0.3...` |
| 41 | 0.7056 | 1 | 7.0e-05 | 0.4 | N/A | False | 0.5 | 0.1 | N/A | N/A | `p1_t012_lr=7e-05_dr=0.40_nacl=0_ul=0.50_nl=0.1...` |
| 42 | 0.7055 | 4 | 1.1e-04 | 0.09 | 9.0e-05 | False | 0.10 | 0.45 | False | 3 | `p4_t006_lr=1e-04_dr=0.09_wd=9e-05_ul=0.10` |
| 43 | 0.7053 | 1 | 7.0e-05 | 0.3 | N/A | False | 0.5 | 0.3 | N/A | N/A | `p1_t013_lr=7e-05_dr=0.30_nacl=0_ul=0.50_nl=0.3...` |
| 44 | 0.7053 | 2 | 1.0e-04 | 0.4 | N/A | False | 0.1 | 0.5 | N/A | N/A | `p2_t006_lr=1e-04_dr=0.40_nacl=0_ul=0.10_nl=0.5...` |
| 45 | 0.7045 | 3 | 1.0e-04 | 0.3 | 1.0e-04 | False | 0.1 | 0.5 | True | 10 | `p3_t003_wd=1e-04_wce=1_T=10_dr=0.30` |
| 46 | 0.7044 | 4 | 1.1e-04 | 0.09 | 1.1e-04 | False | 0.09 | 0.54 | False | 3 | `p4_t002_lr=1e-04_dr=0.09_wd=1e-04_ul=0.10` |
| 47 | 0.7042 | 4 | 1.1e-04 | 0.09 | 1.1e-04 | False | 0.09 | 0.47 | False | 3 | `p4_t001_lr=1e-04_dr=0.09_wd=1e-04_ul=0.10` |
| 48 | 0.7039 | 1 | 5.0e-05 | 0.1 | N/A | False | 0.5 | 0.1 | N/A | N/A | `p1_t017_lr=5e-05_dr=0.10_nacl=0_ul=0.50_nl=0.1...` |
| 49 | 0.7033 | 2 | 1.0e-04 | 0.3 | N/A | False | 0.3 | 0.5 | N/A | N/A | `p2_t008_lr=1e-04_dr=0.30_nacl=0_ul=0.30_nl=0.5...` |
| 50 | 0.7030 | 1 | 5.0e-05 | 0.2 | N/A | True | 0.3 | 0.5 | N/A | N/A | `p1_t018_lr=5e-05_dr=0.20_nacl=1_ul=0.30_nl=0.5...` |
| 51 | 0.6896 | 1 | 1.0e-05 | 0.4 | N/A | True | 0.05 | 0.1 | N/A | N/A | `p1_t014_lr=1e-05_dr=0.40_nacl=1_ul=0.05_nl=0.1...` |
| 52 | 0.6880 | 1 | 1.0e-05 | 0.3 | N/A | True | 0.05 | 0.5 | N/A | N/A | `p1_t016_lr=1e-05_dr=0.30_nacl=1_ul=0.05_nl=0.5...` |

*(Note: Parameters marked N/A were held constant at their default values during that specific phase)*
