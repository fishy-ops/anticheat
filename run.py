"""python run.py [--quick]  ->  reports/results.json, results.md, roc.png, ttd.png, evidence_samples.json"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anticheat.evaluate import CHEATS, FPRS, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    kw = dict(n_train=100, n_cal=600, n_test_legit=300, n_cheat=80) if a.quick else {}
    res, ttd = evaluate(seed=a.seed, workers=min(8, os.cpu_count() or 1), **kw)
    out = Path("reports")

    fm = lambda v: "–" if v is None else f"{v:.1f}"
    L = ["# Anti-cheat results", "",
         f"Calibration legit sessions: {res['config']['n_cal']} · test legit: {res['config']['n_test_legit']} · "
         f"cheaters per type: {res['config']['n_cheat']}", ""]
    for f in FPRS:
        r = res["fpr"][str(f)]
        L += [f"## Target FPR {f:.1%} (measured on held-out legit: {r['_test_fpr']:.2%})", "",
              "| cheat | detection rate | TTD median (s) | TTD p90 (s) | ≤10 s | ≤30 s |", "|---|---|---|---|---|---|"]
        for c in CHEATS:
            x = r[c]
            L.append(f"| {c} | {x['detection_rate']:.1%} | {fm(x['ttd_median_s'])} | {fm(x['ttd_p90_s'])} | "
                     f"{x['detected_within_10s']:.1%} | {x['detected_within_30s']:.1%} |")
        L += ["", "False positives by legit type: " + ", ".join(f"{k} {v:.2%}" for k, v in r["_fp_by_legit_kind"].items()), ""]
    (out / "results.md").write_text("\n".join(L))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in CHEATS:
        ax.plot([p["fpr_target"] for p in res["roc"]], [p[c] for p in res["roc"]], marker="o", ms=3, label=c)
    ax.set_xscale("log"); ax.set_xlabel("session false-positive rate"); ax.set_ylabel("detection rate")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "roc.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in CHEATS:
        v = np.sort(ttd[c])
        if len(v):
            n_all = res["config"]["n_cheat"]
            ax.step(v, np.arange(1, len(v) + 1) / n_all, where="post", label=c)
    ax.set_xlabel("seconds after cheat onset"); ax.set_ylabel("cumulative detection rate (FPR=1%)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "ttd.png", dpi=150)
    print("\n".join(L))


if __name__ == "__main__":
    main()
