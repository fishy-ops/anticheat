"""Calibration + metrics.

Combined score at time t = max over detectors of (running-max CUSUM / scale_det), where scale_det is the
legit-calibration 99th percentile of that detector's final score, so both detectors share one threshold.
The threshold tau is chosen on a CALIBRATION set of legit sessions to hit the target session-level
false-positive rate, then applied unchanged to a disjoint TEST set (different seed).

Metrics (per cheat type):
  * detection rate at fixed FPR  = P(flag any time in session | cheater)
  * time to detection            = flag_time - cheat_onset, over detected cheaters (median / p90)
  * detected within N seconds    = P(flag within N s of onset)
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .detect import AimModel, analyse, evidence
from .sim import make_session

FPRS = [0.001, 0.01, 0.05]
CHEATS = ["aimbot_blatant", "aimbot_subtle", "speed_blatant", "speed_subtle"]
LEGIT_MIX = [("legit", 0.7), ("legit_laggy", 0.2), ("legit_pro", 0.1)]

_MODEL = None


def _legit_kinds(n, seed):
    rng = np.random.default_rng(seed)
    return rng.choice([k for k, _ in LEGIT_MIX], size=n, p=[p for _, p in LEGIT_MIX])


def _init(model):
    global _MODEL
    _MODEL = model


def _job(args):
    sid, kind, seed = args
    return analyse(make_session(sid, kind, seed), _MODEL)


def run_jobs(jobs, model, workers):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(model,)) as ex:
        return list(ex.map(_job, jobs, chunksize=16))


# ----------------------------------------------------------------------------- scoring helpers
def combined_trace(res, scales):
    """(t, score) merged on a common time axis; score = max_det runmax/scale."""
    ts = np.concatenate([res[d]["t"] for d in ("aim", "speed")])
    order = np.argsort(ts, kind="stable")
    sc = np.concatenate([res[d]["runmax"] / scales[d] for d in ("aim", "speed")])[order]
    return ts[order], np.maximum.accumulate(sc)


def flag_time(res, tau, scales):
    ts, sc = combined_trace(res, scales)
    i = np.searchsorted(sc, tau, side="left")
    return None if i >= len(ts) else float(ts[i])


def calibrate(cal, fpr, scales):
    peak = np.array([combined_trace(r, scales)[1][-1] for r in cal])
    return float(np.quantile(peak, 1 - fpr, method="higher"))


def _flagging_detector(r, tau, scales, f):
    """Which detector's normalised score crossed tau first (for evidence)."""
    best = None
    for d in ("aim", "speed"):
        sc = r[d]["runmax"] / scales[d]
        i = np.searchsorted(sc, tau, side="left")
        if i < len(sc) and (best is None or r[d]["t"][i] < best[1]):
            best = (d, float(r[d]["t"][i]))
    return best[0]


def evaluate(n_train=400, n_cal=4000, n_test_legit=1500, n_cheat=400, seed=7, workers=8, out="reports"):
    out = Path(out)
    out.mkdir(exist_ok=True)

    # 1. fit legit population model on a training split
    train = [make_session(i, k, seed) for i, k in enumerate(_legit_kinds(n_train, seed))]
    model = AimModel.fit(train)

    # 2. calibration split (seed+1)
    cal = run_jobs([(i, k, seed + 1) for i, k in enumerate(_legit_kinds(n_cal, seed + 1))], model, workers)
    scales = {d: float(np.quantile([r[d]["runmax"][-1] for r in cal], 0.99)) or 1.0 for d in ("aim", "speed")}

    # 3. test split (seed+2): fresh legit + each cheat type
    jobs = [(i, k, seed + 2) for i, k in enumerate(_legit_kinds(n_test_legit, seed + 2))]
    for ci, c in enumerate(CHEATS):
        jobs += [(10_000 * (ci + 1) + i, c, seed + 2) for i in range(n_cheat)]
    test = run_jobs(jobs, model, workers)
    legit = [r for r in test if r["label"] == 0]

    results = dict(config=dict(n_cal=n_cal, n_test_legit=n_test_legit, n_cheat=n_cheat, seed=seed),
                   scales=scales, fpr={})
    ttd_store, evidence_samples = {}, {}
    for fpr in FPRS:
        tau = calibrate(cal, fpr, scales)
        rows = {"_tau": tau}
        rows["_test_fpr"] = sum(flag_time(r, tau, scales) is not None for r in legit) / len(legit)
        for c in CHEATS:
            rs = [r for r in test if r["kind"] == c]
            fts = [flag_time(r, tau, scales) for r in rs]
            det = [(f, r) for f, r in zip(fts, rs) if f is not None]
            ttd = np.array([max(0.0, f - r["onset"]) for f, r in det])   # pre-onset flags clamp to 0
            within = lambda n: float(np.mean([f is not None and f - r["onset"] <= n for f, r in zip(fts, rs)]))
            rows[c] = dict(detection_rate=len(det) / len(rs),
                           ttd_median_s=float(np.median(ttd)) if len(ttd) else None,
                           ttd_p90_s=float(np.quantile(ttd, 0.9)) if len(ttd) else None,
                           detected_within_10s=within(10), detected_within_30s=within(30))
            if fpr == 0.01:
                ttd_store[c] = ttd.tolist()
                evidence_samples[c] = [
                    dict(sid=r["sid"], kind=c, onset_s=r["onset"], flag_t=f,
                         detector=(d := _flagging_detector(r, tau, scales, f)),
                         evidence=evidence(r, d, f))
                    for f, r in det[:3]]
        rows["_fp_by_legit_kind"] = {
            k: sum(flag_time(r, tau, scales) is not None for r in legit if r["kind"] == k)
               / max(1, sum(r["kind"] == k for r in legit))
            for k, _ in LEGIT_MIX}
        results["fpr"][str(fpr)] = rows

    results["roc"] = []
    for f in np.geomspace(0.0005, 0.2, 14):
        tau = calibrate(cal, f, scales)
        row = dict(fpr_target=float(f))
        for c in CHEATS:
            rs = [r for r in test if r["kind"] == c]
            row[c] = float(np.mean([flag_time(r, tau, scales) is not None for r in rs]))
        results["roc"].append(row)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "evidence_samples.json").write_text(json.dumps(evidence_samples, indent=2))
    return results, ttd_store
