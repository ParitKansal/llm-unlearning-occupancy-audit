# Kill experiment v1: results (run 2026-10-05, Colab A100 ×4 workers + 8-core CPU analysis)

**Pre-registered verdict: NO-GO.** Rule (d) failed: ψ̂ (Mh) on the retain-only model = 0.081 > 0.05. Even ignoring (d),
only 1 of 3 checkpoints passed (a)+(b)+(c) (2 needed), so the result is not GO under any reading of the rule.
Thresholds were not changed after seeing results.

## Raw output
Calibration: false-positive rate per probe 0.022–0.050. Precondition: full-model direct detection 0.708 (PASS).

| Model | M0 ψ̂ [95% CI] | **Mh ψ̂ [95% CI]** (primary) | Mh2 ψ̂ [95% CI] | GOF p (Mh) | single probe | any-of-K |
|---|---|---|---|---|---|---|
| full | 0.808 [0.768, 0.850] | **0.914** [0.881, 0.956] | 0.869 [0.816, 0.918] | 0.337 | 0.708 | 0.925 |
| retain | 0.026 [0.010, 0.043] | **0.081** [0.021, 0.117] | 0.035 [0.012, 0.051] | 0.01 | 0.038 | 0.160 |
| GradDiff | 0.472 [0.435, 0.519] | **0.654** [0.580, 0.751] | 0.563 [0.492, 0.637] | 0.564 | 0.380 | 0.720 |
| NPO | 0.098 [0.008, 0.144] | **0.251** [0.085, 0.342] | 0.111 [0.069, 0.195] | 0.01 | 0.075 | 0.370 |
| RMU | 0.269 [0.227, 0.323] | **0.320** [0.267, 0.396] | 0.293 [0.254, 0.349] | 0.03 | 0.193 | 0.530 |

Relearning (fine-tune on the other author half, all 10 probes): recovery GradDiff 0.735, NPO 0.355, RMU 0.767, retain 0.120.

| Model | ψ\* (truth) | Mh ψ̂ | single | any-of-K | (a) CI covers | (b) beats naive by 2× | (c) single ≤ ψ\*/2 |
|---|---|---|---|---|---|---|---|
| GradDiff | 0.615 | 0.654 | 0.380 | 0.720 | ✅ | ✅ | ❌ |
| NPO | 0.235 | 0.251 | 0.075 | 0.370 | ✅ | ✅ | ✅ |
| RMU | 0.647 | 0.320 | 0.193 | 0.530 | ❌ | ❌ | ✅ |

(d) retain-only ψ̂ ≤ 0.05: ❌ (0.081).

## What we learned (post hoc; exploratory, not confirmatory)
1. **The motivating effect is real and large.** Relearning truth is 1.6× (GradDiff), 3.1× (NPO) and 3.4× (RMU) the single-probe rate.
2. **Where probes can see the knowledge, the estimator works.** GradDiff and NPO: |ψ̂ − ψ\*| = 0.039 and 0.016, versus 0.235 / 0.160 for the single probe and 0.105 / 0.135 for any-of-K.
3. **Failure 1: false positives are clustered, not independent.** On the retain-only model, any-of-K is 0.16, while independent false positives at the measured rates would give ≈ 0.30. Some facts are guessable from the question, and every probe "detects" them. The model reads these as stored-but-hard-to-detect, which inflates ψ̂ where little is stored (retain 0.081) and breaks goodness of fit (p = 0.01 on retain and NPO, 0.03 on RMU).
4. **Failure 2 (structural): RMU hides knowledge from all black-box probes but not from relearning.** ψ\* = 0.65, while every probe-based estimate is ≤ 0.53 and Mh gives 0.32. No black-box estimator can count knowledge that no probe detects; the "detectably stored" estimand is a different quantity from "recoverable by fine-tuning" for representation-scrambling methods like RMU.

## Caveats
One model size, one checkpoint per method, 400 facts; ψ\* is itself a lower bound and uses the retain-only model as its baseline.
