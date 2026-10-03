| Condition | 2AFC acc [95% CI] | n | p vs 50% | Δ vs no_plan [95% CI] | q (BH) | abstain | 'answer 1' rate | mean rank (chance 8) | top-1 | literal mention | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no intervention | 90.0 [86.9, 93.1] | 420 | 1.1e-68 | — | — | 0.00 | 0.55 | 2.70 | 0.62 | 0.01 | 6.23 |
| ablate random dir | 89.5 [86.2, 92.6] | 420 | 8.1e-67 | -0.5 [-5.0, +4.0] | 0.877 | 0.00 | 0.56 | 2.78 | 0.61 | 0.00 | 6.04 |
| ablate other-concept dir | 86.2 [82.6, 89.5] | 420 | 8.5e-55 | -3.8 [-8.6, +1.0] | 0.166 | 0.00 | 0.57 | 3.09 | 0.58 | 0.01 | 3.31 |
| ablate secret dir | 67.1 [61.9, 72.1] | 420 | 1.8e-12 | -22.9 [-28.8, -16.9] | 0 | 0.00 | 0.60 | 5.12 | 0.33 | 0.02 | 3.47 |
| ablate other-concept subspace (14-d) | 80.7 [76.7, 84.8] | 420 | 1.3e-38 | -9.3 [-14.5, -4.0] | 0.0008 | 0.00 | 0.57 | 3.69 | 0.51 | 0.01 | 4.11 |
| ablate secret subspace (14-d) | 60.5 [55.7, 65.2] | 420 | 2e-05 | -29.5 [-35.2, -23.8] | 0 | 0.00 | 0.69 | 6.97 | 0.09 | 0.00 | 3.80 |
| ablate secret dir, first 100 tokens only | 87.1 [81.4, 92.4] | 210 | 1.1e-29 | -2.9 [-9.5, +3.3] | 0.457 | 0.00 | 0.55 | 3.45 | 0.59 | 0.00 | 5.91 |
| ablate secret dir, after token 100 only | 80.0 [73.3, 86.2] | 210 | 4.8e-19 | -10.0 [-17.4, -2.9] | 0.0126 | 0.00 | 0.59 | 4.12 | 0.34 | 0.01 | 3.73 |
| attention knockout of secret sentence | 68.6 [61.4, 75.2] | 210 | 7.7e-08 | -21.4 [-29.0, -14.0] | 0 | 0.00 | 0.65 | 6.03 | 0.19 | 0.07 | 6.58 |

Quality-controlled (only stories judged >= 5/10):

| Condition | # stories >=5 | 2AFC acc on good pairs [95% CI] | trials | mean rank (good) |
|---|---|---|---|---|
| no intervention | 118 | 90.0 [86.8, 93.0] | 400 | 2.67 |
| ablate random dir | 120 | 89.5 [86.2, 92.6] | 420 | 2.78 |
| ablate other-concept dir | 13 | 100.0 [100.0, 100.0] | 8 | 1.31 |
| ablate secret dir | 12 | 100.0 [100.0, 100.0] | 4 | 2.42 |
| ablate other-concept subspace (14-d) | 30 | 67.9 [53.6, 85.7] | 28 | 3.70 |
| ablate secret subspace (14-d) | 16 | 50.0 [50.0, 50.0] | 4 | 6.81 |
| ablate secret dir, first 100 tokens only | 118 | 87.3 [81.4, 92.6] | 204 | 3.49 |
| attention knockout of secret sentence | 120 | 68.6 [61.4, 75.2] | 210 | 6.03 |


OLS per story: rank of true word ~ treatment + judged quality (positive coef = less leakage):

| Comparison | coef [95% CI] | p | quality coef | n |
|---|---|---|---|---|
| abl_secret_vs_abl_other | +2.14 [+1.15, +3.14] | 3.3e-05 | -0.71 | 239 |
| abl_secret_sub_vs_abl_other_sub | +3.11 [+2.00, +4.23] | 9.8e-08 | -0.55 | 240 |
| abl_secret_vs_abl_random | -0.74 [-2.66, +1.18] | 0.45 | -1.20 | 240 |


Steering (no secret in prompt; add r_W at one layer): rank of W among 15

| Condition | mean rank | top-1 | top-3 | n |
|---|---|---|---|---|
| steer_L24_c1 | 7.92 | 0.07 | 0.21 | 120 |
| steer_L24_c3 | 7.73 | 0.07 | 0.24 | 120 |
| no_secret | 7.84 | 0.07 | 0.19 | 120 |
