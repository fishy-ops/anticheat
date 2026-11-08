import numpy as np

from anticheat.detect import AimModel, analyse, cusum_trace
from anticheat.sim import V_MAX, make_session


def test_cusum_resets_and_grows():
    S = cusum_trace(np.array([0, 0, 5, 5, -3, 0.0]), k=1.0)
    assert S[1] == 0 and S[3] == 8 and S[4] == 4 and S[5] == 3


def test_legit_speed_stays_near_cap():
    s = make_session(1, "legit", 0)
    w = 60  # 1 s window, as the detector uses
    chord = np.linalg.norm(s.pos[w:] - s.pos[:-w], axis=1)   # metres per second
    assert chord.max() < V_MAX * 1.1


def test_cheaters_score_higher_than_legit():
    model = AimModel.fit([make_session(i, "legit", 1) for i in range(30)])
    def peak(kind, det):
        return np.mean([analyse(make_session(i, kind, 2), model)[det]["runmax"][-1] for i in range(20)])
    assert peak("aimbot_blatant", "aim") > 3 * peak("legit", "aim")
    assert peak("speed_blatant", "speed") > 3 * peak("legit", "speed")
