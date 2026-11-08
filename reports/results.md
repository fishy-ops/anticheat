# Anti-cheat results

Calibration legit sessions: 4000 · test legit: 1500 · cheaters per type: 400

## Target FPR 0.1% (measured on held-out legit: 0.07%)

| cheat | detection rate | TTD median (s) | TTD p90 (s) | ≤10 s | ≤30 s |
|---|---|---|---|---|---|
| aimbot_blatant | 100.0% | 3.4 | 11.1 | 86.8% | 100.0% |
| aimbot_subtle | 71.2% | 34.4 | 73.5 | 5.2% | 31.0% |
| speed_blatant | 100.0% | 2.2 | 7.3 | 94.8% | 99.8% |
| speed_subtle | 98.8% | 11.3 | 34.3 | 47.2% | 85.2% |

False positives by legit type: legit 0.00%, legit_laggy 0.00%, legit_pro 0.78%

## Target FPR 1.0% (measured on held-out legit: 0.93%)

| cheat | detection rate | TTD median (s) | TTD p90 (s) | ≤10 s | ≤30 s |
|---|---|---|---|---|---|
| aimbot_blatant | 100.0% | 3.4 | 11.1 | 86.8% | 100.0% |
| aimbot_subtle | 92.0% | 20.3 | 57.8 | 22.2% | 60.8% |
| speed_blatant | 100.0% | 1.7 | 6.5 | 96.8% | 100.0% |
| speed_subtle | 100.0% | 5.8 | 21.5 | 69.0% | 95.2% |

False positives by legit type: legit 0.57%, legit_laggy 0.31%, legit_pro 5.43%

## Target FPR 5.0% (measured on held-out legit: 4.87%)

| cheat | detection rate | TTD median (s) | TTD p90 (s) | ≤10 s | ≤30 s |
|---|---|---|---|---|---|
| aimbot_blatant | 100.0% | 3.1 | 11.1 | 86.8% | 100.0% |
| aimbot_subtle | 98.0% | 16.3 | 50.1 | 31.8% | 74.5% |
| speed_blatant | 100.0% | 1.4 | 6.4 | 97.2% | 100.0% |
| speed_subtle | 100.0% | 4.5 | 15.4 | 75.2% | 97.8% |

False positives by legit type: legit 4.19%, legit_laggy 3.42%, legit_pro 13.95%
