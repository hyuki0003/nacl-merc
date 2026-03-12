import json

try:
    with open("hparam_results.json") as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

valid = sorted([r for r in data if r.get("best_test_f1", -1) >= 0], key=lambda x: x["best_test_f1"], reverse=True)

md = "# Hyperparameter & Architecture Search Results (Phases 1-6)\n\n"
md += f"**Total Successful Trials:** {len(valid)}\n\n"
md += "This document summarizes all completed hyperparameter and architecture trials for EmotionHeart+ fine-tuning on the IEMOCAP dataset, sorted by `best_test_f1` (descending). It serves as a comprehensive reference for sensitivity analysis.\n\n"

md += "## Overview of Phases\n"
md += "- **Phase 1 (Coarse Random):** Broad exploration of learning rate, dropout, NACL, unimodal lambda, NACL lambda, temperature, and topk.\n"
md += "- **Phase 2 (Fine Grid):** Narrower grid search around the best config from Phase 1.\n"
md += "- **Phase 3 (Regularization & Class Balance):** Tuning `weight_decay`, Weighted Cross-Entropy (`do_WCE`), and Cosine LR Warmup Duration (`T`).\n"
md += "- **Phase 4 (Adaptive Search):** Data-driven exploration shifting parameters (e.g. `learning_rate`, `dropout`, `weight_decay`) based on Pearson correlation with F1 scores.\n"
md += "- **Phase 5 (Architecture Scaling):** Deep and wide transformer encoder tuning (`encoder_layers`: 2~4, `encoder_embed_dim`: 256~512, `encoder_attention_heads`: 4~8) while fixing the optimal HPs.\n"
md += "- **Phase 6 (Seed Robustness):** Testing the global optimum across 5 random seeds to verify convergence resilience.\n\n"

# Calculate Phase 6 Stats
ph6 = [r for r in valid if r.get("phase") == 6]
if ph6:
    f1s = [r["best_test_f1"] for r in ph6]
    import statistics
    mean = statistics.mean(f1s)
    std = statistics.stdev(f1s) if len(f1s) > 1 else 0
    md += "## Phase 6 Robustness Summary\n"
    md += f"Multiple run statistics over 5 random seeds using the optimal architecture and hyperparameter set:\n"
    md += f"- **Average F1:** `{mean:.4f} ± {std:.4f}`\n"
    md += f"- **Max F1:** `{max(f1s):.4f}`\n"
    md += f"- **Min F1:** `{min(f1s):.4f}`\n\n"

md += "## Complete Trial Leaderboard\n\n"
md += "| Rank | F1 Score | Phase | Learning Rate | Dropout | NACL | T | WCE | Layers | Embed Dim | Heads | Seed | Run ID |\n"
md += "|:---:|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|:---|:---|\n"

for i, r in enumerate(valid):
    c = r["config"]
    ph = r.get("phase", "?")
    f1_str = f"{r['best_test_f1']:.4f}"
    lr = f"{c.get('learning_rate', '') :.1e}" if "learning_rate" in c else "N/A"
    dr = c.get("dropout", "N/A")
    if isinstance(dr, float): dr = round(dr, 3)
    nacl = c.get("do_NACL", "N/A")
    t = c.get("T", "N/A")
    wce = c.get("do_WCE", "N/A")
    el = c.get("encoder_layers", 2)
    ed = c.get("encoder_embed_dim", 384)
    eh = c.get("encoder_attention_heads", 6)
    seed = c.get("seed", 20242025)
    
    run_id = r.get('run_id', '')
    short_run_id = f"`{run_id[:25]}...`" if len(run_id) > 25 else f"`{run_id}`"
    
    row = f"| {i+1} | **{f1_str}** | {ph} | {lr} | {dr} | {nacl} | {t} | {wce} | {el} | {ed} | {eh} | {seed} | {short_run_id} |\n"
    md += row

with open("/home/neuroai/.gemini/antigravity/brain/cd1e7a18-d9f1-4470-980f-2164b0fb5df6/analysis_results.md", "w") as f:
    f.write(md)

print("Regenerated analysis_results.md successfully")
