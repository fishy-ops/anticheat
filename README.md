# Anti-cheat detector

Simulates legit and cheating players, flags cheaters from server-side telemetry, and attaches evidence to every flag.

```
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib pytest
.venv/bin/python run.py            # full run (~5 s on 8 cores) -> reports/
.venv/bin/python run.py --quick
.venv/bin/python -m pytest -q tests
```

## Design
- `anticheat/sim.py` — 60 Hz, 150 s sessions. Legit: casual / good / pro skill, laggy players (freeze + catch-up). Cheats: `aimbot_blatant`, `aimbot_subtle` (partial, smoothed assist), `speed_blatant`, `speed_subtle`. Cheat onset is random (20–60 % into the session).
- `anticheat/detect.py` — two streaming detectors, each a one-sided CUSUM over z-scores. **Speed:** 1 s window path speed vs server cap, lag-compensated. **Aim:** per-engagement reaction time and shot error vs the legit population (log-normal fit). Evidence = the events in the CUSUM excursion before the flag (speed windows with measured m/s vs limit; engagements with reaction ms, shot error, peak flick speed).
- `anticheat/evaluate.py` — one shared threshold on normalised score. **Calibrated on a legit calibration split to a target session-level FPR, applied unchanged to a disjoint test split** (different seed). Reported FPR is re-measured on held-out legit.

## Metrics
- Detection rate at fixed FPR (0.1 %, 1 %, 5 %), per cheat type
- Time to detection = flag time − cheat onset (median, p90, share detected within 10 s / 30 s)
- Pro players and laggy players are included as hard negatives; FPs are broken out by legit type.

Outputs: `reports/results.md`, `results.json`, `roc.png`, `ttd.png`, `evidence_samples.json`.

## Caveats
Everything is synthetic; the numbers measure the detector against *this* simulator's cheat models, not real players. The legit_pro group drives most false positives (a real-world analogue of banning good players).
