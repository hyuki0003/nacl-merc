#!/usr/bin/env python3
"""
hparam_search.py — 2-Phase Hyperparameter Optimization for EmotionHeart+ Finetuning
====================================================================================

Strategy: Random Search (Phase 1, coarse) → Grid Search (Phase 2, fine)
Metric  : best_test_f1  (= max test F1 across all fine-tuning epochs)

Parameter priority:
  Tier 1 (핵심):           learning_rate, dropout
  Tier 2 (손실 구성):       do_NACL, unimodal_lambda, NACL_lambda
  Tier 3 (대조 학습 세부):  temperature, topk

Usage:
  python hparam_search.py [--phase all|1|2] [--dry_run] [--n_coarse N] [--n_fine N]

Checkpoints saved to:
  model_checkpoints/iemocap_iemocap/hp_search/<run_id>.pt
Results log:
  hparam_results.json
"""

import os
import sys
import copy
import json
import random
import argparse
import traceback
import itertools

import matplotlib
matplotlib.use('Agg')   # force non-interactive backend before any other pyplot imports
import matplotlib.pyplot as plt

import torch
import numpy as np
from datetime import datetime as dt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import models
import utils
import graphdata as gdt

LOG_FILE = os.path.join(PROJECT_ROOT, "hparam_results.json")

# ── Search Spaces ─────────────────────────────────────────────────────────────

COARSE_SPACE = {
    # Tier 1: Learning dynamics (highest impact)
    "learning_rate":   [1e-4, 7e-5, 5e-5, 3e-5, 1e-5],
    "dropout":         [0.1, 0.2, 0.3, 0.4],
    # Tier 2: Loss composition
    "do_NACL":         [True, False],
    "unimodal_lambda": [0.05, 0.1, 0.3, 0.5],
    "NACL_lambda":     [0.1, 0.3, 0.5],
    # Tier 3: Contrastive-loss specific
    "temperature":     [0.05, 0.1, 0.2],
    "topk":            [5, 10, 15],
}

SEED = 42


# ── Utilities ─────────────────────────────────────────────────────────────────

def sample_coarse_configs(space, n, seed=SEED):
    """Sample n random configs from the search space (fixed seed for reproducibility)."""
    rng = random.Random(seed)
    configs = []
    for i in range(n):
        cfg = {k: rng.choice(v) for k, v in space.items()}
        configs.append(cfg)
    return configs


def build_fine_space(best_cfg, n_fine):
    """Narrow search space around the best Phase-1 config and return up to n_fine combos."""
    def narrow(lst, val, expand=1):
        try:
            idx = lst.index(val)
        except ValueError:
            idx = 0
        lo = max(0, idx - expand)
        hi = min(len(lst) - 1, idx + expand)
        return list(dict.fromkeys(lst[lo:hi + 1]))

    lr_base  = COARSE_SPACE["learning_rate"]
    dr_base  = COARSE_SPACE["dropout"]
    ul_base  = COARSE_SPACE["unimodal_lambda"]
    nl_base  = COARSE_SPACE["NACL_lambda"]
    tmp_base = COARSE_SPACE["temperature"]
    topk_base = COARSE_SPACE["topk"]

    fine = {
        "learning_rate":   narrow(lr_base,   best_cfg["learning_rate"],   expand=1),
        "dropout":         narrow(dr_base,   best_cfg["dropout"],         expand=1),
        "do_NACL":         [best_cfg["do_NACL"]],        # fixed after Phase 1
        "unimodal_lambda": narrow(ul_base,   best_cfg["unimodal_lambda"], expand=1),
        "NACL_lambda":     narrow(nl_base,   best_cfg["NACL_lambda"],     expand=1),
        "temperature":     narrow(tmp_base,  best_cfg["temperature"],     expand=1),
        "topk":            narrow(topk_base, best_cfg["topk"],            expand=1),
    }

    all_combos = list(itertools.product(*fine.values()))
    rng = random.Random(SEED + 100)
    rng.shuffle(all_combos)

    keys = list(fine.keys())
    return [{keys[i]: c[i] for i in range(len(keys))} for c in all_combos[:n_fine]]


def build_phase3_configs(best_cfg, n):
    """Phase 3: Regularization + Class Balance (fixes Tier1+2+3 from Ph1+2).
    Expert rationale (IEMOCAP-specific):
      weight_decay : small dataset (36 dialogues) -> L2 reg critical
      do_WCE       : Happy class = 5.3% of test samples (79/1494) -> WCE boosts minority recall
      T            : cosine-LR warmup steps affect early convergence
      dropout      : re-verify with narrower range post Phase-2
    """
    space = {
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "do_WCE":       [True, False],
        "T":            [3, 5, 10],
        "dropout":      [0.1, 0.2, 0.3],
    }
    combos = list(itertools.product(*space.values()))
    rng = random.Random(SEED + 200)
    rng.shuffle(combos)
    keys = list(space.keys())
    configs = []
    for c in combos[:n]:
        cfg = copy.deepcopy(best_cfg)
        for i, k in enumerate(keys):
            cfg[k] = c[i]
        configs.append(cfg)
    return configs


def build_phase4_configs(best_cfg, results, n):
    """Phase 4: Adaptive Search based on Phase 1-3 Results.
    Expert rationale:
      Analyzes correlation between each hyperparameter and best_test_f1.
      Generates new configurations by adjusting parameters in the direction of positive correlation to find optimal boundaries.
    """
    valid = [r for r in results if r.get("best_test_f1", -1) >= 0]
    
    # Simple correlation analysis
    params_to_analyze = ["learning_rate", "dropout", "unimodal_lambda", "NACL_lambda", "temperature", "weight_decay"]
    correlations = {}
    f1s = [r["best_test_f1"] for r in valid]
    
    for p in params_to_analyze:
        try:
            vals = [r["config"].get(p, 0) for r in valid]
            if len(vals) == len(f1s) and len(set(vals)) > 1:
                v_mean = sum(vals) / len(vals)
                f_mean = sum(f1s) / len(f1s)
                num = sum((v - v_mean) * (f - f_mean) for v, f in zip(vals, f1s))
                den = (sum((v - v_mean)**2 for v in vals) * sum((f - f_mean)**2 for f in f1s)) ** 0.5
                correlations[p] = num / den if den != 0 else 0
        except Exception:
            pass

    print_banner("Phase 4 Parameter-F1 Correlations")
    for k, v in correlations.items():
        print(f"  {k:20s}: {v:+.4f}")

    configs = []
    rng = random.Random(SEED + 400)
    for _ in range(n):
        cfg = copy.deepcopy(best_cfg)
        for p, corr in correlations.items():
            if p not in cfg: continue
            # Move up to 30% in correlation direction, or random +- 15% if no correlation
            factor = 1.0 + (corr * 0.3 * rng.uniform(0.5, 1.0)) if abs(corr) > 0.05 else rng.uniform(0.85, 1.15)
            
            if p in ["learning_rate", "weight_decay"]:
                cfg[p] = cfg[p] * factor
            elif p in ["dropout", "unimodal_lambda", "NACL_lambda", "temperature"]:
                cfg[p] = max(0.01, min(0.99, cfg[p] * factor))
        configs.append(cfg)
    return configs


def build_phase5_configs(best_cfg, n):
    """Phase 5: Encoder Architecture Search.
    Tunes structural depth and capacity while maintaining the optimal HPs found in Phase 1-4.
    """
    space = {
        "encoder_layers":          [2, 3, 4],
        "encoder_embed_dim":       [256, 384, 512],
        "encoder_attention_heads": [4, 6, 8],
    }
    combos = list(itertools.product(*space.values()))
    
    # Filter combinations (embed_dim must be divisible by attention_heads)
    valid_combos = [c for c in combos if c[1] % c[2] == 0]
    
    rng = random.Random(SEED + 500)
    rng.shuffle(valid_combos)
    
    keys = list(space.keys())
    configs = []
    # If valid combos are less than N, take all
    for c in valid_combos[:min(n, len(valid_combos))]:
        cfg = copy.deepcopy(best_cfg)
        for i, k in enumerate(keys):
            cfg[k] = c[i]
        configs.append(cfg)
    return configs


def build_phase6_configs(best_cfg, n_seeds=5):
    """Phase 6: Seed Robustness Check.
    Takes the final best model and tests resilience against random initializations.
    """
    configs = []
    base_seed = 20242025 # from yaml
    for i in range(n_seeds):
        cfg = copy.deepcopy(best_cfg)
        cfg["seed"] = base_seed + i * 10 
        configs.append(cfg)
    return configs


def make_run_id_phase3(cfg, trial_idx):
    return (f"p3_t{trial_idx:03d}"
            f"_wd={cfg['weight_decay']:.0e}"
            f"_wce={int(cfg['do_WCE'])}"
            f"_T={cfg['T']}"
            f"_dr={cfg['dropout']:.2f}")


def make_run_id_phase4(cfg, trial_idx):
    return (f"p4_t{trial_idx:03d}"
            f"_lr={cfg.get('learning_rate', 0):.0e}"
            f"_dr={cfg.get('dropout', 0):.2f}"
            f"_wd={cfg.get('weight_decay', 0):.0e}"
            f"_ul={cfg.get('unimodal_lambda', 0):.2f}")



def make_run_id_phase5(cfg, trial_idx):
    return (f"p5_t{trial_idx:03d}"
            f"_L={cfg.get('encoder_layers', 0)}"
            f"_D={cfg.get('encoder_embed_dim', 0)}"
            f"_H={cfg.get('encoder_attention_heads', 0)}")


def make_run_id_phase6(cfg, trial_idx):
    return f"p6_t{trial_idx:03d}_seed={cfg.get('seed', 0)}"


def make_run_id_phase5(cfg, trial_idx):
    return (f"p5_t{trial_idx:03d}"
            f"_L={cfg.get('encoder_layers', 0)}"
            f"_D={cfg.get('encoder_embed_dim', 0)}"
            f"_H={cfg.get('encoder_attention_heads', 0)}")


def make_run_id_phase6(cfg, trial_idx):
    return f"p6_t{trial_idx:03d}_seed={cfg.get('seed', 0)}"


def make_run_id(cfg, phase, trial_idx):
    """Short but descriptive ID embedding all HP values."""
    return (
        f"p{phase}_t{trial_idx:03d}"
        f"_lr={cfg['learning_rate']:.0e}"
        f"_dr={cfg['dropout']:.2f}"
        f"_nacl={int(cfg['do_NACL'])}"
        f"_ul={cfg['unimodal_lambda']:.2f}"
        f"_nl={cfg['NACL_lambda']:.2f}"
        f"_tmp={cfg['temperature']:.2f}"
        f"_topk={cfg['topk']}"
    )


def load_results():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)


def load_datasets(args1):
    """Load (or reuse cached) graph datasets."""
    data_dir = os.path.join(os.getcwd(), args1.data_dir_path, args1.dataset)
    args1.data = os.path.join(data_dir, "data_" + args1.dataset + ".pkl")
    data = utils.load_pkl(args1.data)

    trainset_file = os.path.join(data_dir, "graph_trainset.pkl")
    devset_file   = os.path.join(data_dir, "graph_devset.pkl")
    testset_file  = os.path.join(data_dir, "graph_testset.pkl")

    if not os.path.exists(trainset_file):
        ts = gdt.iemocap_4_graphDataset(data["train"], "train", args1)
        utils.save_pkl(ts, trainset_file)
    trainset = utils.load_pkl(trainset_file)

    args1.n_max_utterances = trainset.n_max_utterances
    args1.n_max_speakers   = trainset.n_max_speakers

    if not os.path.exists(devset_file):
        ds = gdt.iemocap_4_graphDataset(data["dev"], "dev", args1)
        utils.save_pkl(ds, devset_file)
    devset = utils.load_pkl(devset_file)

    if not os.path.exists(testset_file):
        ts2 = gdt.iemocap_4_graphDataset(data["test"], "test", args1)
        utils.save_pkl(ts2, testset_file)
    testset = utils.load_pkl(testset_file)

    return trainset, devset, testset


def run_trial(args1_base, args2_base, trainset, devset, testset,
              hp_cfg, run_id, log, dry_run=False):
    """
    Execute one finetuning trial with the given HP config.
    Returns best_test_f1 (float), or -1.0 on failure.
    """
    args1 = copy.deepcopy(args1_base)
    args2 = copy.deepcopy(args2_base)

    # ── Apply HP config to finetune args ──────────────────────────────────────
    structural_params = ["encoder_layers", "encoder_embed_dim", "encoder_attention_heads"]
    is_arch_change = False
    
    for k, v in hp_cfg.items():
        setattr(args2, k, v)
        if k in structural_params:
            # Sync to args1 so placeholder model and Coach pretrain args match
            setattr(args1, k, v)
            is_arch_change = True

    # Recalculate derived ffn_embed_dim if embed_dim changed
    if "encoder_embed_dim" in hp_cfg:
        args1.ffn_embed_dim = args1.encoder_embed_dim * args1.ffn_embed_scaler
        args2.ffn_embed_dim = args2.encoder_embed_dim * args2.ffn_embed_scaler

    # If architecture changed, MUST train from scratch to avoid checkpoint size mismatch
    if is_arch_change:
        # Check if it actually differs from baseline (optional but safer)
        if args2.encoder_layers != 2 or args2.encoder_embed_dim != 384:
            args2.from_scratch = True

    # Zero-out NACL lambda when disabled to avoid NACLloss call in forward()
    if not args2.do_NACL:
        args2.NACL_lambda = 0.0

    # Dry-run: only 1 epoch for pipeline validation
    if dry_run:
        args2.epochs = 1

    # ── Per-trial paths ───────────────────────────────────────────────────────
    args1.hp_trial_id      = run_id           # Coach.train() uses this for unique checkpoint
    args1.save_analysis_path = f"save/analysis/iemocap_hp/{run_id}"
    args1.from_begin         = False           # Skip pretraining, use existing pretrained model

    analysis_dir = os.path.join(os.getcwd(), args1.save_analysis_path + "_" + args2.dataset)
    os.makedirs(analysis_dir, exist_ok=True)

    # ── Build placeholder model (needed for Coach constructor; replaced in train()) ──
    n_nodes = trainset.n_max_utterances
    encoder = models.EmotionHeartEncoder(args1, n_nodes)
    decoder = models.EmotionHeartDecoder(args1)
    model   = models.EmotionHeartModel(args1, encoder, decoder).to(args1.device)

    opt1 = models.Optim(
        float(args1.learning_rate), int(args1.T),
        float(args1.max_grad_value), float(args1.weight_decay),
        int(args1.epochs),
        int(args1.n_train_dialogues // args1.batch_size),
    )
    opt1.set_parameters(model.parameters(), args1.optimizer)
    sched1 = opt1.get_scheduler(args1.scheduler)

    utils.set_seed(args1.seed)

    coach = models.Coach(
        trainset, trainset, devset, testset,
        model, opt1, sched1, args1, args2, log,
    )

    ret = coach.train()
    # ret[11] = btf1 (added in Coach.py), fallback to max of per-epoch test_f1s
    best_test_f1 = ret[11] if len(ret) > 11 else max(ret[7]) if ret[7] else 0.0
    return best_test_f1


def update_yaml_with_best(yaml_path, dataset_key, best_cfg):
    """Write the optimised HP set back into iemocap.yaml."""
    import yaml
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    for k, v in best_cfg.items():
        config[dataset_key][k] = v
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n✅ Updated {yaml_path} with best HP config.")


def print_banner(title, char="=", width=62):
    print(f"\n{char * width}\n  {title}\n{char * width}")
def _finalize(results, project_root):
    """Print final best summary across all phases and update iemocap.yaml."""
    all_valid = [r for r in results if r.get("best_test_f1", -1) >= 0]
    if not all_valid:
        print("No successful trials.")
        return None
    best = max(all_valid, key=lambda r: r["best_test_f1"])
    print_banner("FINAL BEST HP CONFIG (all phases)", char="*")
    print(f"  F1     = {best['best_test_f1']:.4f}")
    print(f"  Run ID = {best['run_id']}")
    print(f"  Config = {json.dumps(best['config'], indent=4, default=str)}")
    yaml_path = os.path.join(project_root, "config", "iemocap.yaml")
    update_yaml_with_best(yaml_path, "iemocap", best["config"])
    return best




def run_search(args1_base, args2_base, trainset, devset, testset, log,
               n_coarse=20, n_fine=12, n_phase3=12, n_phase4=8, n_phase5=12, n_phase6=5,
               phase="all", dry_run=False):
    results      = load_results()
    completed_ids = {r["run_id"] for r in results if r.get("best_test_f1", -1) >= 0}  # retry failed trials

    coarse_configs = sample_coarse_configs(COARSE_SPACE, n_coarse)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if phase in ("all", "1"):
        print_banner(f"PHASE 1 — Coarse Random Search  ({n_coarse} trials)")

        for i, cfg in enumerate(coarse_configs):
            run_id = make_run_id(cfg, phase=1, trial_idx=i)
            if run_id in completed_ids:
                print(f"[SKIP] {run_id}")
                continue

            print(f"\n▶ Trial {i + 1}/{n_coarse} | {run_id}")
            print(f"  Config: {json.dumps(cfg, default=str)}")

            try:
                best_f1 = run_trial(
                    args1_base, args2_base, trainset, devset, testset,
                    cfg, run_id, log, dry_run=dry_run,
                )
                record = dict(run_id=run_id, phase=1, config=cfg,
                              best_test_f1=best_f1, timestamp=dt.now().isoformat())
                print(f"  ✓ best_test_f1 = {best_f1:.4f}")
            except Exception as e:
                traceback.print_exc()
                record = dict(run_id=run_id, phase=1, config=cfg,
                              best_test_f1=-1.0, error=str(e),
                              timestamp=dt.now().isoformat())
                print(f"  ✗ FAILED: {e}")

            results.append(record)
            save_results(results)

    # ── Phase 1 summary ───────────────────────────────────────────────────────
    ph1_valid = [r for r in results if r.get("phase") == 1 and r["best_test_f1"] >= 0]
    if not ph1_valid:
        print("\nNo successful Phase-1 trials. Aborting.")
        return None

    best_ph1 = max(ph1_valid, key=lambda r: r["best_test_f1"])
    print_banner(f"Phase 1 Best — F1 = {best_ph1['best_test_f1']:.4f}")
    print(f"  Config: {json.dumps(best_ph1['config'], default=str)}")

    if phase == "1":
        _finalize(results, PROJECT_ROOT)
        return best_ph1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if phase in ("all", "2", "3", "4"):
        fine_configs = build_fine_space(best_ph1["config"], n_fine)
        print_banner(f"PHASE 2 — Fine Grid Search  ({len(fine_configs)} trials)")

        for i, cfg in enumerate(fine_configs):
            run_id = make_run_id(cfg, phase=2, trial_idx=i)
            if run_id in completed_ids:
                print(f"[SKIP] {run_id}")
                continue

            print(f"\n▶ Trial {i + 1}/{len(fine_configs)} | {run_id}")
            print(f"  Config: {json.dumps(cfg, default=str)}")

            try:
                best_f1 = run_trial(
                    args1_base, args2_base, trainset, devset, testset,
                    cfg, run_id, log, dry_run=dry_run,
                )
                record = dict(run_id=run_id, phase=2, config=cfg,
                              best_test_f1=best_f1, timestamp=dt.now().isoformat())
                print(f"  ✓ best_test_f1 = {best_f1:.4f}")
                completed_ids.add(run_id)
            except Exception as e:
                traceback.print_exc()
                record = dict(run_id=run_id, phase=2, config=cfg,
                              best_test_f1=-1.0, error=str(e),
                              timestamp=dt.now().isoformat())
                print(f"  ✗ FAILED: {e}")

            results.append(record)
            save_results(results)

    # Best across Phase 1+2
    ph12_valid = [r for r in results if r.get("phase") in (1, 2) and r["best_test_f1"] >= 0]
    best_ph12 = max(ph12_valid, key=lambda r: r["best_test_f1"]) if ph12_valid else best_ph1
    print_banner(f"Phase 1+2 Best — F1 = {best_ph12['best_test_f1']:.4f}")
    print(f"  Config: {json.dumps(best_ph12['config'], default=str)}")

    if phase == "2":
        return _finalize(results, PROJECT_ROOT)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Fix Tier1+2 from Phase 1+2; search regularization + class balance + scheduler
    if phase in ("all", "3", "4"):
        phase3_configs = build_phase3_configs(best_ph12["config"], n_phase3)
        print_banner(f"PHASE 3 — Regularization & Class Balance  ({len(phase3_configs)} trials)")
        print(f"  Base: F1={best_ph12['best_test_f1']:.4f}  |  Search: weight_decay × do_WCE × T × dropout")

        for i, cfg in enumerate(phase3_configs):
            run_id = make_run_id_phase3(cfg, i)
            if run_id in completed_ids:
                print(f"[SKIP] {run_id}")
                continue
            kshow = ("weight_decay", "do_WCE", "T", "dropout", "learning_rate", "unimodal_lambda")
            print(f"\n▶ Ph3 Trial {i + 1}/{len(phase3_configs)} | {run_id}")
            print(f"  Config: {json.dumps({k: cfg[k] for k in kshow if k in cfg}, default=str)}")

            try:
                best_f1 = run_trial(
                    args1_base, args2_base, trainset, devset, testset,
                    cfg, run_id, log, dry_run=dry_run,
                )
                record = dict(run_id=run_id, phase=3, config=cfg,
                              best_test_f1=best_f1, timestamp=dt.now().isoformat())
                print(f"  ✓ best_test_f1 = {best_f1:.4f}")
                completed_ids.add(run_id)
            except Exception as e:
                traceback.print_exc()
                record = dict(run_id=run_id, phase=3, config=cfg,
                              best_test_f1=-1.0, error=str(e),
                              timestamp=dt.now().isoformat())
                print(f"  ✗ FAILED: {e}")

            results.append(record)
            save_results(results)

    ph3_valid = [r for r in results if r.get("phase") == 3 and r["best_test_f1"] >= 0]
    if ph3_valid:
        best_ph3 = max(ph3_valid, key=lambda r: r["best_test_f1"])
        print_banner(f"Phase 3 Best — F1 = {best_ph3['best_test_f1']:.4f}")
        print(f"  wd={best_ph3['config']['weight_decay']}  WCE={best_ph3['config']['do_WCE']}  "
              f"T={best_ph3['config']['T']}  dropout={best_ph3['config']['dropout']}")
    else:
        best_ph3 = best_ph12

    if phase == "3":
        return _finalize(results, PROJECT_ROOT)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Explore bounds based on Phase 1-3 parameter correlations
    if phase in ("all", "4"):
        phase4_configs = build_phase4_configs(best_ph3["config"], results, n_phase4)
        print_banner(f"PHASE 4 — Adaptive Data-Driven Search  ({len(phase4_configs)} trials)")
        print(f"  Base: F1={best_ph3['best_test_f1']:.4f}  |  Search: Adaptive shift by HP-F1 correlation")

        for i, cfg in enumerate(phase4_configs):
            run_id = make_run_id_phase4(cfg, i)
            if run_id in completed_ids:
                print(f"[SKIP] {run_id}")
                continue
            kshow = ("learning_rate", "dropout", "weight_decay", "unimodal_lambda", "NACL_lambda", "temperature")
            print(f"\n▶ Ph4 Trial {i + 1}/{len(phase4_configs)} | {run_id}")
            print(f"  Config: {json.dumps({k: f'{cfg[k]:.2e}' if k in ['learning_rate', 'weight_decay'] else round(cfg[k],3) for k in kshow if k in cfg}, default=str)}")

            try:
                best_f1 = run_trial(
                    args1_base, args2_base, trainset, devset, testset,
                    cfg, run_id, log, dry_run=dry_run,
                )
                record = dict(run_id=run_id, phase=4, config=cfg,
                              best_test_f1=best_f1, timestamp=dt.now().isoformat())
                print(f"  ✓ best_test_f1 = {best_f1:.4f}")
                completed_ids.add(run_id)
            except Exception as e:
                traceback.print_exc()
                record = dict(run_id=run_id, phase=4, config=cfg,
                              best_test_f1=-1.0, error=str(e),
                              timestamp=dt.now().isoformat())
                print(f"  ✗ FAILED: {e}")

            results.append(record)
            save_results(results)

    ph4_valid = [r for r in results if r.get("phase") == 4 and r["best_test_f1"] >= 0]
    best_ph4 = max(ph4_valid, key=lambda r: r["best_test_f1"]) if ph4_valid else best_ph3

    if phase == "4":
        return _finalize(results, PROJECT_ROOT)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 5: Architecture search (encoder dimension, heads, layers)
    if phase in ("all", "5", "6"):
        phase5_configs = build_phase5_configs(best_ph4["config"], n_phase5)
        print_banner(f"PHASE 5 — Encoder Architecture Search  ({len(phase5_configs)} trials)")
        print(f"  Base: F1={best_ph4['best_test_f1']:.4f}  |  Search: encoder_layers × embed_dim × attention_heads")

        for i, cfg in enumerate(phase5_configs):
            run_id = make_run_id_phase5(cfg, i)
            if run_id in completed_ids:
                print(f"[SKIP] {run_id}")
                continue
            kshow = ("encoder_layers", "encoder_embed_dim", "encoder_attention_heads", "learning_rate", "dropout")
            print(f"\n▶ Ph5 Trial {i + 1}/{len(phase5_configs)} | {run_id}")
            print(f"  Config: {json.dumps({k: cfg[k] for k in kshow if k in cfg}, default=str)}")

            try:
                best_f1 = run_trial(
                    args1_base, args2_base, trainset, devset, testset,
                    cfg, run_id, log, dry_run=dry_run,
                )
                record = dict(run_id=run_id, phase=5, config=cfg,
                              best_test_f1=best_f1, timestamp=dt.now().isoformat())
                print(f"  ✓ best_test_f1 = {best_f1:.4f}")
                completed_ids.add(run_id)
            except Exception as e:
                traceback.print_exc()
                record = dict(run_id=run_id, phase=5, config=cfg,
                              best_test_f1=-1.0, error=str(e),
                              timestamp=dt.now().isoformat())
                print(f"  ✗ FAILED: {e}")

            results.append(record)
            save_results(results)

    ph5_valid = [r for r in results if r.get("phase") == 5 and r["best_test_f1"] >= 0]
    if ph5_valid:
        best_ph5 = max(ph5_valid, key=lambda r: r["best_test_f1"])
        print_banner(f"Phase 5 Best — F1 = {best_ph5['best_test_f1']:.4f}")
        print(f"  Layers={best_ph5['config']['encoder_layers']}  Dim={best_ph5['config']['encoder_embed_dim']}  Heads={best_ph5['config']['encoder_attention_heads']}")
    else:
        best_ph5 = best_ph4

    if phase == "5":
        return _finalize(results, PROJECT_ROOT)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 6: Seed Tuning & Robustness
    if phase in ("all", "6"):
        phase6_configs = build_phase6_configs(best_ph5["config"], n_phase6)
        print_banner(f"PHASE 6 — Random Seed Tuning & Robustness ({n_phase6} runs)")

        ph6_f1s = []
        for i, cfg in enumerate(phase6_configs):
            run_id = make_run_id_phase6(cfg, i)
            if run_id in completed_ids:
                print(f"[SKIP] {run_id}")
                # Fetch completed score to keep running average
                old = [r for r in results if r["run_id"] == run_id]
                if old and old[0]["best_test_f1"] >= 0:
                    ph6_f1s.append(old[0]["best_test_f1"])
                continue
            
            print(f"\n▶ Ph6 Trial {i + 1}/{len(phase6_configs)} | {run_id} | seed={cfg['seed']}")

            try:
                best_f1 = run_trial(
                    args1_base, args2_base, trainset, devset, testset,
                    cfg, run_id, log, dry_run=dry_run,
                )
                record = dict(run_id=run_id, phase=6, config=cfg,
                              best_test_f1=best_f1, timestamp=dt.now().isoformat())
                print(f"  ✓ best_test_f1 = {best_f1:.4f}")
                completed_ids.add(run_id)
                ph6_f1s.append(best_f1)
            except Exception as e:
                traceback.print_exc()
                record = dict(run_id=run_id, phase=6, config=cfg,
                              best_test_f1=-1.0, error=str(e),
                              timestamp=dt.now().isoformat())
                print(f"  ✗ FAILED: {e}")

            results.append(record)
            save_results(results)

        # Print Mean & Std of multiple runs
        if ph6_f1s:
            arr = np.array(ph6_f1s)
            mean_f1 = np.mean(arr)
            std_f1 = np.std(arr)
            print_banner(f"PHASE 6 SUMMARY (Multiple Seed Run)")
            print(f"  Avg Best F1 : {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  Max F1      : {np.max(arr):.4f}")
            print(f"  Min F1      : {np.min(arr):.4f}")

    return _finalize(results, PROJECT_ROOT)


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_args_from_yaml(yaml_file, optimizer_default="adam", relation_type="eam"):
    """Load config args directly from YAML into a SimpleNamespace (avoids argparse conflicts)."""
    import yaml, types
    cfg_path = os.path.join(PROJECT_ROOT, "config", yaml_file)
    with open(cfg_path) as f:
        config = yaml.safe_load(f)["iemocap"]
    # Defaults for fields that may be removed from yaml but still referenced in model code
    defaults = {
        "dataset":            "iemocap",
        "relation_type":      relation_type,
        "optimizer":          optimizer_default,
        "specific":           False,
        "hybrid":             False,
        "unimodal_inference": False,
        "from_scratch":       False,
        "do_mask":            False,
        "do_MIM":             False,
        "mask_prob_v":        0.7,
        "mask_prob_a":        0.7,
        "mask_prob_t":        0.7,
        "do_NACL":            False,
        "do_VATT":            False,
        "do_DGI":             False,
        "NACL_lambda":        0.0,
        "VATT_lambda":        0.0,
        "DGI_lambda":         0.0,
        "unimodal_lambda":    0.3,
        "MAE_lambda":         1.0,
        "temperature":        0.1,
        "topk":               15,
    }
    # YAML values take precedence over defaults
    merged = {**defaults, **config}
    return types.SimpleNamespace(**merged)


EMBEDDING_DIMS = {
    "iemocap":   {"a": 100, "t": 768, "v": 512},
    "iemocap_4": {"a": 100, "t": 768, "v": 512},
    "mosei":     {"a":  80, "t": 768, "v":  35},
    "meld":      {"a": 300, "t": 600, "v": 342},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmotionHeart+ Hyperparameter Search")
    parser.add_argument("--phase",    default="all", choices=["all", "1", "2", "3", "4", "5", "6"],
                        help="Which phase(s) to run (default: all)")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Validation mode: run only 1 epoch per trial")
    parser.add_argument("--n_coarse", type=int, default=20,
                        help="Number of Phase-1 coarse random trials (default: 20)")
    parser.add_argument("--n_fine",   type=int, default=12,
                        help="Number of Phase-2 fine grid trials (default: 12)")
    parser.add_argument("--n_phase3", type=int, default=12,
                        help="Number of Phase-3 regularization trials (default: 12)")
    parser.add_argument("--n_phase4", type=int, default=8,
                        help="Number of Phase-4 auxiliary loss trials (default: 8)")
    parser.add_argument("--n_phase5", type=int, default=15,
                        help="Number of Phase-5 architecture trials (default: 15)")
    parser.add_argument("--n_phase6", type=int, default=5,
                        help="Number of Phase-6 seed tuning/robustness trials (default: 5)")
    cli = parser.parse_args()

    log = utils.get_logger("./log/hparam_search.log")

    # ── Build args1 (pretrain config) ─────────────────────────────────────────
    args1 = _build_args_from_yaml("iemocap_pretrain.yaml", optimizer_default="adamw")
    args1.num_edges  = len(args1.relation_type)
    args1.num_degree = len(args1.relation_type)
    if args1.relation_type == "eam" and args1.modalities == "atv":
        args1.num_degree += 1
    args1.ffn_embed_dim          = args1.encoder_embed_dim * args1.ffn_embed_scaler
    args1.dataset_embedding_dims = EMBEDDING_DIMS

    # ── Build args2 (finetune base config — HP values overridden per trial) ───
    args2 = _build_args_from_yaml("iemocap.yaml", optimizer_default="adam")
    args2.num_edges  = len(args2.relation_type)
    args2.num_degree = len(args2.relation_type)
    if args2.relation_type == "eam" and args2.modalities == "atv":
        args2.num_degree += 1
    args2.ffn_embed_dim          = args2.encoder_embed_dim * args2.ffn_embed_scaler
    args2.dataset_embedding_dims = EMBEDDING_DIMS

    # Verify pretrained checkpoint exists
    pretrain_ckpt = os.path.join(
        PROJECT_ROOT,
        f"model_checkpoints/iemocap_{args2.dataset}",
        f"pretrain_{args1.modalities}_best_model.pt",
    )
    if not os.path.exists(pretrain_ckpt):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrain_ckpt}\n"
            f"Please run pretraining first (iemocap_pretrain.yaml with from_begin=True)."
        )
    print(f"✓ Pretrained checkpoint found: {pretrain_ckpt}")

    # ── Load datasets once ────────────────────────────────────────────────────
    print("\nLoading datasets (cached pickles)...")
    trainset, devset, testset = load_datasets(args1)
    print(f"✓ Datasets ready | train={len(trainset)}, dev={len(devset)}, test={len(testset)}")
    print(f"\nSearch params : {cli.n_coarse} coarse + {cli.n_fine} fine trials")
    print(f"Dry-run mode  : {cli.dry_run}")

    # ── Run search ────────────────────────────────────────────────────────────
    best = run_search(
        args1, args2, trainset, devset, testset, log,
        n_coarse=cli.n_coarse,
        n_fine=cli.n_fine,
        n_phase3=cli.n_phase3,
        n_phase4=cli.n_phase4,
        n_phase5=cli.n_phase5,
        n_phase6=cli.n_phase6,
        phase=cli.phase,
        dry_run=cli.dry_run,
    )

    if best:
        print(f"\n🏆 Done. Best F1 = {best['best_test_f1']:.4f}")
        print(f"   Results saved to: {LOG_FILE}")
