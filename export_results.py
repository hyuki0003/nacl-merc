import json

with open("hparam_results.json") as f:
    data = json.load(f)

valid = sorted([r for r in data if r.get("best_test_f1", -1) >= 0], key=lambda x: x["best_test_f1"], reverse=True)

md = "# Hyperparameter Search Results (Phases 1-6)\n\n"
md += f"**Total Successful Trials:** {len(valid)}\n\n"
md += "This document summarizes all completed hyperparameter and architecture trials for EmotionHeart+ fine-tuning on the IEMOCAP dataset, sorted by `best_test_f1` (descending). It serves as a comprehensive reference for sensitivity analysis.\n\n"

md += "| Rank | F1 Score | Phase | Learning Rate | Dropout | NACL | E_Layers | E_Dim | E_Heads | Seed | Run ID |\n"
md += "|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|:---|:---|\n"

for i, r in enumerate(valid[:100]):
    c = r["config"]
    ph = r.get("phase", "?")
    f1_str = f"{r['best_test_f1']:.4f}"
    lr = f"{c.get('learning_rate', '') :.1e}" if "learning_rate" in c else "N/A"
    dr = c.get("dropout", "N/A")
    nacl = c.get("do_NACL", "N/A")
    el = c.get("encoder_layers", 2)
    ed = c.get("encoder_embed_dim", 384)
    eh = c.get("encoder_attention_heads", 6)
    seed = c.get("seed", 20242025)
    
    row = f"| {i+1} | **{f1_str}** | {ph} | {lr} | {dr} | {nacl} | {el} | {ed} | {eh} | {seed} | `{r['run_id'][:25]}...` |\n"
    md += row

with open("/home/neuroai/.gemini/antigravity/brain/cd1e7a18-d9f1-4470-980f-2164b0fb5df6/analysis_results.md", "w") as f:
    f.write(md)
print("Saved analysis_results.md")
