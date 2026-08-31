# [Under Review] Cross-Modal Alignment for Robust Multimodal Fusion in Conversational Emotion Recognition

<div align="center">

**Dae Hyeon Kim**<sup>a</sup>, **Dong-Hyuk Lee**<sup>a,b</sup>, **Young-Seok Choi**<sup>a,*</sup>

<sup>a</sup>Department of Electronics and Communications Engineering, Kwangwoon University, Seoul, Republic of Korea
<br>
<sup>b</sup>Medical AI Co., Ltd., Seoul, Republic of Korea

[![Journal](https://img.shields.io/badge/Expert%20Systems%20with%20Applications-Under%20Review-orange.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 📢 News
* **[Aug. 2026]** 🚀 The official code released!
* **[Aug. 2026]** 📝 Our paper **"Cross-Modal Alignment for Robust Multimodal Fusion in Conversational Emotion Recognition"** was submitted to **Expert Systems with Applications** on **August 11, 2026** and is currently **under review**.

---

## 📝 Abstract

As artificial intelligence moves toward empathetic interaction, understanding emotion in conversation has become essential for affective human-computer interaction. In conversation, emotion is conveyed through heterogeneous facial, vocal, and linguistic cues shaped by speaker relations and context dynamics, making Multimodal Emotion Recognition in Conversation (MERC) a challenging problem. Although recent MERC studies have advanced contextual aggregation and multimodal fusion, explicit alignment of heterogeneous modality representations remains underexplored. Consequently, cross-modal embeddings remain distributionally inconsistent, exacerbating modality gap and modality collapse, reducing robustness to missing modalities, and increasing vulnerability to domain shift.

To address this issue, we propose a self-supervised heterogeneous graph representation learning framework that combines a multimodal masked autoencoder (**MMAE**) with Neighbor Alignment Contrastive Learning (**NACL**), a contrastive objective for cross-modal neighborhood alignment. Under a strict cross-dataset transfer setting, experiments show that explicit neighbor alignment improves representation quality and MERC performance, while MMAE and NACL provide complementary benefits. Crucially, we demonstrate that explicit geometric alignment acts as a structural regularizer that mitigates unimodal over-reliance, thereby facilitating a more balanced integration of complementary modalities. These results support cross-modal neighborhood alignment as an effective objective for multimodal representation learning in MERC.

**Framework at a glance.** Each dialogue is modeled as a heterogeneous graph with one node per modality (audio / text / visual) per utterance, connected by intra-speaker, inter-speaker, and inter-modality relations. A unified Heterogeneous Graph Transformer encoder (Graphormer-style) encodes it via Entity Encoding (position, speaker, modality, degree embeddings) and Structure Encoding (shortest-path spatial bias and edge-relation attribute bias). The encoder is pretrained with `L_pretrain = L_mse (MMAE) + λ_con · L_con (NACL)`; for transfer, the decoder is discarded and a linear-fusion classifier is fine-tuned with `L_sup + λ_con · L_con`, i.e., NACL also serves as an auxiliary alignment regularizer.

---

## 🗂️ Repository Structure

```
.
├── main_crossdataset.py        # Stage 1: SSL pretraining on source (MELD) -> Stage 2: fine-tuning on target (IEMOCAP)
├── main_intradataset.py        # Train-from-scratch or pretrain+finetune on a single dataset (MELD / IEMOCAP)
├── config/
│   ├── meld_pretrain.yaml         # MELD SSL pretraining (MMAE + NACL) — 4-way transfer setting
│   ├── meld_pretrain_6way.yaml    # MELD SSL pretraining — 6-way transfer setting (mask 0.7, k 3, tau 0.7)
│   ├── meld_pretrain_scratch.yaml # Stage-1 config for MELD from-scratch (4-layer encoder)
│   ├── iemocap_pretrain.yaml      # IEMOCAP intra-dataset pretraining
│   ├── iemocap_4.yaml             # IEMOCAP 4-way fine-tuning / evaluation
│   ├── iemocap.yaml               # IEMOCAP 6-way fine-tuning / evaluation
│   └── meld.yaml                  # MELD intra-dataset (from-scratch) training
├── preprocessing/              # Data schemas and split scripts (see preprocessing/README.md)
├── data/                       # Dataset loaders; feature pickles are NOT shipped (see Data Preparation)
├── graphdata/                  # Heterogeneous graph construction + Floyd-Warshall spatial encoding
├── models/
│   ├── Coach.py                # Trainer (pretraining, fine-tuning, evaluation, checkpointing)
│   ├── Optim.py                # Optimizer / LR scheduler wrapper
│   └── emotionheart/           # HGT encoder (Graphormer-style), MMAE decoder, NACL loss, fine-tune wrapper
├── utils.py
└── requirements.txt
```

Datasets (`data/`), checkpoints (`model_checkpoints/`), and experiment outputs (`save/`, `log/`) are not part of the repository and are created or supplied locally.

---

## ⚙️ Environment

- Python 3.9
- PyTorch 2.7.1 + CUDA (tested with CUDA 12.x)
- fairseq 0.12.2 (required by the Graphormer-based encoder)

```bash
pip install "pip<24.1"   # fairseq 0.12.2 -> omegaconf<2.1 wheels carry legacy metadata rejected by pip >= 24.1
pip install -r requirements.txt
```

The paper's experiments were run on Ubuntu 22.04.3 with 3× NVIDIA RTX 3090 GPUs. A single GPU is sufficient to run the code (`device` in the configs, or `--device` on the command line).

---

## 📁 Data Preparation

Features must be prepared from the official [IEMOCAP](https://sail.usc.edu/iemocap/) and [MELD](https://affective-meld.github.io/) releases; the per-utterance features follow the standard MERC feature extraction used in prior work (e.g., COGMEN / MMGCN-style features). Place the pickles as follows:

| File | Format |
| --- | --- |
| `data/iemocap/data_iemocap.pkl` | dict `{train, dev, test}` of dialogues; per utterance: audio 100-d (openSMILE), text 768-d (SBERT), visual 512-d; 6-way labels |
| `data/iemocap_4/data_iemocap_4.pkl` | same layout, 4-way labels (hap/sad/neu/ang) |
| `data/meld/MELD_features_raw1.pkl` | 10-element list: videoIDs, speakers, labels, text 600-d, audio 300-d, visual 342-d, sentences, train/dev/test ids |
| `data/meld/data_meld.pkl` | same layout, with the paper's split indices |

See `preprocessing/README.md` for the exact schemas and the split tooling (`iemocap_split.py` reproduces the fixed 108/12/31 dialogue split; `meld_split.py` searches for a balanced MELD split).

On the first run, the main scripts build heterogeneous graph datasets (including Floyd-Warshall shortest-path spatial encodings) and cache them as `data/<dataset>/graph_*set.pkl`. This can take a while and the caches are large; subsequent runs load them directly.

---

## 🚀 Training

```bash
# Cross-dataset transfer: MELD pretraining -> IEMOCAP 4-way fine-tuning (main result)
python main_crossdataset.py --pretrain_dataset meld --finetune_dataset iemocap_4

# Cross-dataset transfer: MELD pretraining -> IEMOCAP 6-way fine-tuning
python main_crossdataset.py --pretrain_dataset meld --finetune_dataset iemocap \
    --pretrain_config config/meld_pretrain_6way.yaml

# MELD from scratch (supervised + NACL regularizer; 4-layer encoder from the stage-1 config)
python main_intradataset.py --dataset meld --pretrain_config config/meld_pretrain_scratch.yaml
```

The encoder architecture is always taken from the stage-1 (`*_pretrain`) config — including from-scratch runs, where stage 1 is skipped but still supplies the encoder definition.

Stage behavior is controlled by the config files — `config/<dataset>_pretrain.yaml` for stage 1 and `config/<dataset>.yaml` for stage 2:

| Key | Stage | Effect |
| --- | --- | --- |
| `from_begin` | pretrain | `true`: run self-supervised pretraining; `false`: reuse the saved pretrained checkpoint |
| `do_finetune` | finetune | `true`: run supervised fine-tuning on the target dataset |
| `from_scratch` | finetune | `true`: skip pretraining and train the encoder from random initialization |
| `unimodal_inference` | finetune | `true`: missing-modality inference with the saved fine-tuned model (see Evaluation) |
| `freeze_prefixes` | finetune | Parameter-name prefixes frozen during fine-tuning; the 6-way transfer config freezes the 2nd encoder block (`encoder.graph_encoder.layers.1`), as in the paper; 4-way uses full fine-tuning (`[]`) |

Checkpoints are written to `model_checkpoints/<pretrain>_<target>/pretrain_atv_best_model.pt` (stage 1) and `model_checkpoints/<pretrain>_<target>/finetune_atv_best_model.pt` (stage 2), e.g. `model_checkpoints/meld_iemocap_4/`. Loss curves and metrics are saved under `save/analysis/`.

---

## 🔎 Evaluation

Evaluate a saved fine-tuned checkpoint on the test set:

```bash
python main_crossdataset.py --finetune_dataset iemocap_4 --eval_only
python main_crossdataset.py --finetune_dataset iemocap --eval_only
python main_intradataset.py --dataset meld --eval_only
```

**Missing-modality inference.** To evaluate the full-modality fine-tuned model when a modality is unavailable at test time, set `do_finetune: false` and `unimodal_inference: true` in the target config (`config/iemocap_4.yaml` etc.), and select the kept subset with `--modalities`:

```bash
# e.g. audio + text only (visual missing)
python main_crossdataset.py --finetune_dataset iemocap_4 --modalities at
```

Any subset of `{a, t, v}` is supported (`a`, `t`, `v`, `at`, `tv`, `av`).

---

## 📊 Experimental Results

Weighted F1 (%) from the paper:

| Setting | w-F1 |
| :--- | :---: |
| MELD → IEMOCAP 4-way (transfer) | **84.7** |
| IEMOCAP 4-way (from scratch) | 82.0 |
| MELD → IEMOCAP 6-way (transfer) | **71.9** |
| MELD (from scratch) | **68.8** |

Key hyperparameters (also documented in `config/*.yaml`):

| Hyperparameter | Value |
| --- | --- |
| Hidden dimension | 384 |
| Attention heads | 6 |
| Encoder layers | 2 (MELD from scratch: 4) |
| Mask ratio (MELD pretraining) | 0.5 per modality (4-way) / 0.7 (6-way) |
| NACL `k` / `tau` (IEMOCAP 4-way) | 7 / 0.1 (pretraining and fine-tuning) |
| NACL `k` / `tau` (IEMOCAP 6-way, MELD) | 3 / 0.7 (pretraining and fine-tuning) |
| `lambda_con` | 1.0 (pretraining), 0.5 (fine-tuning) |
| Optimizer | AdamW (pretraining), Adam (fine-tuning) |
| LR schedule | cosine (linear warmup during pretraining) |

---

## Citation

```bibtex
@article{kim2026crossmodal,
  title   = {Cross-Modal Alignment for Robust Multimodal Fusion in Conversational Emotion Recognition},
  author  = {Kim, Dae Hyeon and Lee, Dong-Hyuk and Choi, Young-Seok},
  journal = {Expert Systems with Applications},
  note    = {Submitted August 11, 2026; under review},
  year    = {2026}
}
```

## 🙏 Acknowledgments

The graph encoder builds on [Microsoft Graphormer](https://github.com/microsoft/Graphormer) (MIT License) and uses components from [fairseq](https://github.com/facebookresearch/fairseq). We thank the authors of these projects; the corresponding source files retain their original license headers.
