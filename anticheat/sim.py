"""Gameplay simulator: legit players (casual / good / pro, some laggy) and cheaters.

Two telemetry streams per session, both server-side observable:
  * movement: (x, y) position per tick, plus a `lag` effect (frozen positions then catch-up)
  * engagements: aim-error trace (degrees, 2D) for ~1 s after a target appears, plus the shot tick

Cheat onset is a random point inside the session so time-to-detection is measurable.
Ground truth (`label`, `kind`, `onset_s`) is carried on the Session but never read by detectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TICK_HZ = 60
DT = 1.0 / TICK_HZ
SESSION_S = 150
V_MAX = 6.0            # server-enforced sprint speed, m/s
ENG_TICKS = 60         # aim trace length after target spawn (1 s)

SKILLS = {
    # react: median reaction ms, rs: log-sigma, err: per-axis shot error sigma (deg), w: Fitts width
    "casual": dict(react=290, rs=0.28, err=1.30, w=0.35),
    "good":   dict(react=240, rs=0.24, err=0.80, w=0.30),
    "pro":    dict(react=185, rs=0.20, err=0.35, w=0.25),
}


@dataclass
class Session:
    sid: int
    kind: str                      # legit | legit_laggy | legit_pro | aimbot_blatant | aimbot_subtle | speed_blatant | speed_subtle
    label: int                     # 1 = cheater
    onset_s: float | None
    pos: np.ndarray                # (T, 2) metres, as observed by the server
    eng_t: np.ndarray              # engagement spawn times (s)
    eng_err: np.ndarray            # (N, ENG_TICKS, 2) aim error vector, degrees
    eng_shot: np.ndarray           # (N,) tick index of the shot within the trace
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- movement
def _movement(rng, n_ticks, speed_mult_after=None, onset_tick=None, lag_rate=0.0):
    """Piecewise-constant speed/heading, box-smoothed. Returns observed positions (T,2)."""
    seg = []
    t = 0
    while t < n_ticks:
        L = int(rng.uniform(1.0, 4.0) * TICK_HZ)
        sp = rng.choice([0.0, 3.5, V_MAX], p=[0.25, 0.30, 0.45])
        hd = rng.uniform(0, 2 * np.pi)
        seg.append((L, sp, hd))
        t += L
    sp = np.concatenate([np.full(L, s) for L, s, _ in seg])[:n_ticks]
    hd = np.concatenate([np.full(L, h) for L, _, h in seg])[:n_ticks]
    k = np.ones(12) / 12
    vx = np.convolve(sp * np.cos(hd), k, mode="same")
    vy = np.convolve(sp * np.sin(hd), k, mode="same")
    if speed_mult_after is not None:
        m = np.ones(n_ticks)
        m[onset_tick:] = speed_mult_after
        vx, vy = vx * m, vy * m
    pos = np.cumsum(np.stack([vx, vy], 1) * DT, axis=0)
    pos += rng.normal(0, 0.02, pos.shape)          # measurement jitter

    # lag spikes: server sees frozen position, then a catch-up jump
    n_spikes = rng.poisson(lag_rate * n_ticks / (TICK_HZ * 60))
    for _ in range(n_spikes):
        s = int(rng.integers(TICK_HZ, n_ticks - 40))
        d = int(rng.uniform(0.1, 0.5) * TICK_HZ)
        pos[s:s + d] = pos[s]
    return pos


# ----------------------------------------------------------------------------- aim
def _human_engagement(rng, sk, theta0):
    """Human flick: reaction delay, Fitts-law ballistic move with over/undershoot, noisy settle, shot."""
    mag = np.linalg.norm(theta0)
    react = rng.lognormal(np.log(sk["react"] / 1000), sk["rs"])
    move_t = 0.10 + 0.09 * np.log2(1 + mag / sk["w"] / 3)
    t = np.arange(ENG_TICKS) * DT
    u = np.clip((t - react) / move_t, 0, 1)
    prog = 10 * u**3 - 15 * u**4 + 6 * u**5                     # minimum-jerk profile
    gain = 1 + rng.normal(0.04, 0.05)                           # over/undershoot
    err = theta0[None, :] - theta0[None, :] * (gain * prog)[:, None]
    settle = np.clip((t - react - move_t) / 0.15, 0, 1)
    fin = rng.normal(0, sk["err"], 2) * (0.4 + 0.6 * mag / (mag + 10))
    err = err + (1 - np.exp(-settle * 3))[:, None] * (fin[None, :] - err)
    err += rng.normal(0, 0.03, err.shape)                       # hand tremor
    shot = int(np.clip((react + move_t + rng.uniform(0.04, 0.12)) / DT, 5, ENG_TICKS - 1))
    return err, shot


def _aimbot_engagement(rng, theta0, subtle):
    if subtle:
        react = rng.lognormal(np.log(0.15), 0.30)               # reduced but not instant
        smooth = int(rng.integers(4, 11))                       # ticks to converge
        resid_sigma = rng.uniform(0.2, 0.5)
    else:
        react = rng.lognormal(np.log(0.03), 0.30)
        smooth = int(rng.integers(1, 3))
        resid_sigma = 0.04
    tick0 = int(react / DT)
    err = np.tile(theta0, (ENG_TICKS, 1)).astype(float)
    fin = rng.normal(0, resid_sigma, 2)
    for i in range(tick0, ENG_TICKS):
        f = min(1.0, (i - tick0 + 1) / smooth)
        err[i] = theta0 + f * (fin - theta0)
    err += rng.normal(0, 0.05 if subtle else 0.01, err.shape)
    shot = int(np.clip(tick0 + smooth + rng.integers(1, 4), 3, ENG_TICKS - 1))
    return err, shot


def _aim_stream(rng, skill, cheat=None, onset_s=None):
    sk = SKILLS[skill]
    n_eng = int(SESSION_S / rng.uniform(4.0, 6.0))
    eng_t = np.sort(rng.uniform(3, SESSION_S - 3, n_eng))
    errs, shots = [], []
    for t0 in eng_t:
        mag = rng.uniform(4, 90)
        ang = rng.uniform(0, 2 * np.pi)
        theta0 = mag * np.array([np.cos(ang), np.sin(ang)])
        cheating = cheat is not None and t0 >= onset_s
        if cheating and cheat == "aimbot_blatant":
            e, s = _aimbot_engagement(rng, theta0, False)
        elif cheating and cheat == "aimbot_subtle" and rng.random() < 0.4:   # assists some fights only
            e, s = _aimbot_engagement(rng, theta0, True)
        else:
            e, s = _human_engagement(rng, sk, theta0)
        errs.append(e)
        shots.append(s)
    return eng_t, np.array(errs), np.array(shots)


# ----------------------------------------------------------------------------- sessions
def make_session(sid: int, kind: str, seed: int) -> Session:
    rng = np.random.default_rng([seed, sid])
    n = SESSION_S * TICK_HZ
    is_cheat = kind.startswith(("aimbot", "speed"))
    onset_s = float(rng.uniform(0.2, 0.6) * SESSION_S) if is_cheat else None
    cheat_aim = kind if kind.startswith("aimbot") else None

    if kind == "legit_pro":
        skill = "pro"
    elif kind in ("legit", "legit_laggy"):
        skill = str(rng.choice(["casual", "good"], p=[0.65, 0.35]))
    else:
        skill = str(rng.choice(["casual", "good", "pro"], p=[0.5, 0.35, 0.15]))

    lag_rate = 0.0
    if kind == "legit_laggy":
        lag_rate = rng.uniform(2, 6)               # spikes per minute
    elif rng.random() < 0.15:
        lag_rate = rng.uniform(0.3, 2)

    mult, onset_tick = None, None
    if kind.startswith("speed"):
        mult = rng.uniform(1.45, 2.0) if kind == "speed_blatant" else rng.uniform(1.05, 1.15)
        onset_tick = int(onset_s * TICK_HZ)

    pos = _movement(rng, n, mult, onset_tick, lag_rate)
    eng_t, eng_err, eng_shot = _aim_stream(rng, skill, cheat_aim, onset_s)
    return Session(sid, kind, int(is_cheat), onset_s, pos, eng_t, eng_err, eng_shot,
                   dict(skill=skill, lag_rate=lag_rate, speed_mult=mult))
