"""CPU check of the occupancy estimator on simulated data (no GPU, no model).

Run: python test_occ.py
Passes if, over 20 simulated audits that look like the planned experiment, the 95% CI of psi from the
primary model covers the true psi in at least 85% of runs, and the naive estimators are clearly biased.
"""
import numpy as np
import occ

K = 10
A = np.array([-2.2, -1.8, -1.5, -1.2, -1.0, -0.5, 0.0, 0.5, -2.5, 1.0])   # weak and strong probes
F = np.full(K, 0.02)                                                     # 2% false-positive rate per probe


def one(seed, psi=0.45, sigma=1.0, n=400, B=40):
    Y, _ = occ.simulate(n, psi, A, F, sigma=sigma, seed=seed)
    r = occ.fit(Y, F, "Mh")
    lo, hi, _ = occ.bootstrap_ci(Y, F, "Mh", B=B, seed=seed)
    nv = occ.naive_estimates(Y)
    return r["psi"], lo, hi, nv["single_probe"], nv["any_of_K"]


if __name__ == "__main__":
    psi = 0.45
    rows = np.array([one(s, psi) for s in range(20)])
    cover = np.mean((rows[:, 1] <= psi) & (psi <= rows[:, 2]))
    print(f"true psi {psi}")
    print(f"Mh psi-hat mean {rows[:,0].mean():.3f} (sd {rows[:,0].std():.3f}); 95% CI coverage {cover:.2f}")
    print(f"single-probe mean {rows[:,3].mean():.3f}; any-of-K mean {rows[:,4].mean():.3f}")
    # retain-only control: nothing stored, only false positives (5%, the calibration ceiling)
    F5 = np.full(K, 0.05)
    c = np.array([occ.fit(occ.simulate(400, 0.0, A, F5, seed=s)[0], F5, "Mh")["psi"] for s in range(20)])
    print(f"control (psi=0, f=5%): psi-hat mean {c.mean():.3f}, max {c.max():.3f} over 20 seeds")
    assert cover >= 0.85, cover
    assert abs(rows[:, 0].mean() - psi) < 0.06
    assert c.max() <= 0.05, c.max()
    print("PASS")
