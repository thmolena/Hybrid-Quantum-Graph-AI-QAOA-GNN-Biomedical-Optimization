#!/usr/bin/env python3
"""Generate CSV + LaTeX artifacts for the matched-anchor OOD negative result
from the saved v3 JSON outputs. Deterministic; no re-simulation."""
import json, sys, pathlib, math
import numpy as np

SCRATCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
from exp_ood_v3 import paired_stats, METHODS  # reuse exact stat defs

REPO = pathlib.Path("/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/"
                    "Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization")
TAB = REPO / "submission" / "code" / "tables"

def load(b):
    return json.load(open(SCRATCH / f"ood_v3_b{b}.json"))

def arr(rows, m):
    return np.array([r[m] for r in rows], float)

def regime_rows(data, which):
    if which == "in_dist":
        return data["in_distribution"]
    fam_meta = data["family_meta"]
    fams = data["ood_by_family"]
    if which == "genuine_ood":
        out = []
        for f, rows in fams.items():
            if fam_meta[f]["kind"] == "trivial":
                continue
            out += [r for r in rows if r.get("_is_ood")]
        return out
    if which == "geometric":
        return [r for f, rows in fams.items() if fam_meta[f]["kind"] == "geometric" for r in rows]

COMPS = [
    ("uncertainty_geometry", "uq_qaoa", "anchors_fixed",
     "UQ-QAOA $-$ matched-anchor (isolates uncertainty geometry)"),
    ("vs_physics", "uq_qaoa", "tqa_refine",
     "UQ-QAOA $-$ TQA+refine (full method vs physics)"),
    ("learned_center", "post_fixed", "tqa_refine",
     "Learned center $-$ TQA+refine"),
]

def main():
    csv_lines = ["budget,regime,N,comparison,mean_delta,median_delta,win_rate,wins,losses,cohens_dz,sign_p"]
    # main LaTeX table at Q=18
    data18 = load(18); data30 = load(30)
    tex = []
    tex.append(r"\begin{tabular}{@{}llrrrr@{}}")
    tex.append(r"\toprule")
    tex.append(r"Regime & Isolated comparison & $\Delta$ & win\% & $d_z$ & sign $p$ \\")
    tex.append(r"\midrule")
    regime_label = {"in_dist": "In-distribution", "genuine_ood": "Genuine OOD",
                    "geometric": "Geometric"}
    for b, data in [(18, data18), (30, data30)]:
        for which in ["in_dist", "genuine_ood", "geometric"]:
            rows = regime_rows(data, which)
            for key, a, bm, label in COMPS:
                d = arr(rows, a) - arr(rows, bm)
                st = paired_stats(d)
                csv_lines.append(f"{b},{which},{len(rows)},{key},{st['mean']:.4f},"
                                 f"{st['median']:.4f},{st['win_rate']:.3f},{st['wins']},"
                                 f"{st['losses']},{st['cohens_dz']:.3f},{st['sign_p']:.3g}")
    # Build a compact main table from Q=18 only
    for which in ["in_dist", "genuine_ood", "geometric"]:
        rows = regime_rows(data18, which)
        first = True
        for key, a, bm, label in COMPS:
            d = arr(rows, a) - arr(rows, bm)
            st = paired_stats(d)
            reg = f"{regime_label[which]} ($N{{=}}{len(rows)}$)" if first else ""
            first = False
            sigp = st["sign_p"]
            sigs = f"{sigp:.2g}" if sigp >= 1e-4 else f"$<10^{{{int(math.floor(math.log10(sigp)))}}}$"
            tex.append(f"{reg} & {label} & ${st['mean']:+.3f}$ & "
                       f"${100*st['win_rate']:.0f}$ & ${st['cohens_dz']:+.2f}$ & {sigs} \\\\")
        tex.append(r"\addlinespace")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # per-family table (Q=18)
    fam_csv = ["family,kind,frac_genuine_ood,med_optcut,N,uq_minus_tqa_mean,uq_minus_anchors_mean,center_minus_tqa_mean,center_minus_tqa_winrate,center_minus_tqa_signp"]
    fam_meta = data18["family_meta"]
    for f, rows in data18["ood_by_family"].items():
        m = fam_meta[f]
        d_uq_tqa = paired_stats(arr(rows,"uq_qaoa")-arr(rows,"tqa_refine"))
        d_uq_anc = paired_stats(arr(rows,"uq_qaoa")-arr(rows,"anchors_fixed"))
        d_ctr = paired_stats(arr(rows,"post_fixed")-arr(rows,"tqa_refine"))
        fam_csv.append(f"{f},{m['kind']},{m['frac_ood']:.2f},{m['med_optcut']:.2f},{m['n']},"
                       f"{d_uq_tqa['mean']:.4f},{d_uq_anc['mean']:.4f},{d_ctr['mean']:.4f},"
                       f"{d_ctr['win_rate']:.2f},{d_ctr['sign_p']:.3g}")

    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "ood_matched_anchor_summary.csv").write_text("\n".join(csv_lines) + "\n")
    (TAB / "ood_per_family.csv").write_text("\n".join(fam_csv) + "\n")
    (TAB / "table_ood_matched_anchor.tex").write_text("\n".join(tex) + "\n")
    print("wrote:")
    for p in ["ood_matched_anchor_summary.csv","ood_per_family.csv","table_ood_matched_anchor.tex"]:
        print("  ", TAB / p)
    print("\n--- main table preview ---")
    print("\n".join(tex))
    print("\n--- per-family preview ---")
    print("\n".join(fam_csv))

if __name__ == "__main__":
    main()
