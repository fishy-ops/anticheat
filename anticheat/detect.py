"""Streaming detectors. They see only observable telemetry (pos, aim traces), never labels.

Each detector turns events into z-scores and runs a one-sided CUSUM  S = max(0, S + z - k).
A session is flagged the first time the (normalised) running score crosses a threshold; the
evidence rows that pushed the score up are returned so every flag is explainable.
"""
from __future__ import annotations

import numpy as np

from .sim import DT, ENG_TICKS, TICK_HZ, V_MAX, Session

CUSUM_K = 1.0


# ----------------------------------------------------------------------------- speed
def _lag_compensate(pos: np.ndarray) -> np.ndarray:
    """Per-tick displacement with catch-up jumps spread over the preceding frozen ticks."""
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    out = d.copy()
    frozen = d < 0.03                      # below jitter floor
    for j in np.where(d > 2.5 * V_MAX * DT)[0]:
        i = j
        while i > 0 and frozen[i - 1]:
            i -= 1
        if j - i >= 2:                     # needs a freeze run before the jump
            out[i:j + 1] = d[j] / (j - i + 1)
    return out


def speed_events(s: Session, win_s=1.0, stride_s=0.5):
    d = _lag_compensate(s.pos)
    w, st = int(win_s * TICK_HZ), int(stride_s * TICK_HZ)
    c = np.concatenate([[0], np.cumsum(d)])
    starts = np.arange(0, len(d) - w, st)
    speed = (c[starts + w] - c[starts]) / win_s
    t = (starts + w) * DT
    z = np.clip((speed / V_MAX - 1.0) / 0.03, -1, 12)
    ev = [dict(t=float(a), speed_mps=float(b), limit_mps=V_MAX, ratio=float(b / V_MAX), z=float(c_))
          for a, b, c_ in zip(t, speed, z)]
    return t, z, ev


# ----------------------------------------------------------------------------- aim
def aim_features(s: Session):
    """Per-engagement features computed from the raw aim-error trace."""
    err = s.eng_err                                   # (N, T, 2)
    mag = np.linalg.norm(err, axis=2)
    vel = np.linalg.norm(np.diff(err, axis=1), axis=2) / DT        # deg/s
    moving = vel > 40.0
    react_tick = np.where(moving.any(1), moving.argmax(1) + 1, ENG_TICKS)
    shot = s.eng_shot
    return dict(react_ms=react_tick * DT * 1000,
                shot_err=mag[np.arange(len(shot)), shot],
                peak=vel.max(1),
                initial=mag[:, 0])


class AimModel:
    """Legit-population log-normal fits for reaction and shot error (one-sided 'too good' z-scores)."""

    def __init__(self, mu_r, sd_r, mu_e, sd_e):
        self.mu_r, self.sd_r, self.mu_e, self.sd_e = mu_r, sd_r, mu_e, sd_e

    @classmethod
    def fit(cls, sessions):
        r, e = [], []
        for s in sessions:
            f = aim_features(s)
            r.append(np.log(np.maximum(f["react_ms"], 1.0)))
            e.append(np.log(np.maximum(f["shot_err"], 1e-3)))
        r, e = np.concatenate(r), np.concatenate(e)
        return cls(r.mean(), r.std(), e.mean(), e.std())

    def events(self, s: Session):
        f = aim_features(s)
        zr = (self.mu_r - np.log(np.maximum(f["react_ms"], 1.0))) / self.sd_r
        ze = (self.mu_e - np.log(np.maximum(f["shot_err"], 1e-3))) / self.sd_e
        z = np.clip(0.5 * (zr + ze), -1, 8)
        t = s.eng_t + s.eng_shot * DT
        ev = [dict(t=float(a), react_ms=float(b), shot_err_deg=float(c), peak_deg_s=float(d),
                   initial_offset_deg=float(e_), z=float(g))
              for a, b, c, d, e_, g in zip(t, f["react_ms"], f["shot_err"], f["peak"], f["initial"], z)]
        return t, z, ev


# ----------------------------------------------------------------------------- scoring
def cusum_trace(z: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    S = np.zeros(len(z))
    acc = 0.0
    for i, v in enumerate(z):
        acc = max(0.0, acc + v - k)
        S[i] = acc
    return S


def analyse(s: Session, aim_model: AimModel) -> dict:
    """Compact per-session result: CUSUM traces + evidence rows."""
    out = dict(sid=s.sid, kind=s.kind, label=s.label, onset=s.onset_s)
    for name, (t, z, ev) in dict(aim=aim_model.events(s), speed=speed_events(s)).items():
        S = cusum_trace(z)
        out[name] = dict(t=t, S=S, runmax=np.maximum.accumulate(S), ev=ev, z=z)
    return out


def evidence(res: dict, det: str, flag_t: float, top: int = 5) -> list[dict]:
    """Events in the CUSUM excursion leading up to the flag, highest z first."""
    d = res[det]
    idx = np.where(d["t"] <= flag_t)[0]
    if len(idx) == 0:
        return []
    i0 = idx[-1]
    while i0 > 0 and d["S"][i0 - 1] > 0:
        i0 -= 1
    rows = [d["ev"][i] for i in range(i0, idx[-1] + 1)]
    rows.sort(key=lambda r: -r["z"])
    return rows[:top]
