"""Build kill_experiment.ipynb (self-contained: embeds occ.py and probes.py). Run: python make_notebook.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Kill experiment: occupancy models for unlearning audits (TOFU, Llama-3.2-1B)

**Question.** After unlearning, what fraction of the forget set is still *stored* in the model? One probe failing is a
non-detection, not proof of absence. We probe each fact with 10 probe types ("visits"), fit an occupancy model that
separates ψ (fact still stored) from p (probe detects a stored fact), and compare ψ̂ with a relearning ground truth.

**Pre-registered go / no-go:** see `PREREG.md` (same thresholds are coded in the last cell).

**Runtime:** Colab A100, about 2 GPU-hours. Every stage saves to Google Drive and is skipped on rerun (resume-safe).
Uses public OpenUnlearning checkpoints, so no unlearning training is needed.
""")

code("""
!pip -q install -U transformers accelerate
import os, json, time
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/occupancy_unlearning/kill_v1'
os.makedirs(OUT, exist_ok=True)
print(OUT)
""")

md("## 1. Code (written to files so the notebook is self-contained)")
code("%%writefile occ.py\n" + (HERE / "occ.py").read_text())
code("%%writefile probes.py\n" + (HERE / "probes.py").read_text())

md("""
## 2. Estimator check on simulated data (CPU, ~1 min)
Simulate an audit like ours (400 facts, 10 probes, fact heterogeneity, 2% false positives, true ψ = 0.45).
The primary model should recover ψ; the single-probe and any-of-K rates should be biased.
""")
code("""
import numpy as np, occ
A = np.array([-2.2, -1.8, -1.5, -1.2, -1.0, -0.5, 0.0, 0.5, -2.5, 1.0]); F = np.full(10, 0.02)
for seed in range(3):
    Y, _ = occ.simulate(400, 0.45, A, F, sigma=1.0, seed=seed)
    r = occ.fit(Y, F, 'Mh'); lo, hi, _ = occ.bootstrap_ci(Y, F, 'Mh', B=40, seed=seed)
    print(f"seed {seed}: psi-hat {r['psi']:.3f} [{lo:.3f}, {hi:.3f}]  naive {occ.naive_estimates(Y)}")
""")

md("## 3. Config and data (pre-registered; do not change after the first run)")
code("""
import numpy as np, torch, probes as P
HUB = 'open-unlearning/'
MODELS = {
  'full':     HUB + 'tofu_Llama-3.2-1B-Instruct_full',        # trained on all TOFU facts (before unlearning)
  'retain':   HUB + 'tofu_Llama-3.2-1B-Instruct_retain90',    # never saw forget10: false-positive control
  'GradDiff': HUB + 'unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10',
  'NPO':      HUB + 'unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10',
  'RMU':      HUB + 'unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10',
}
UNLEARNED = ['GradDiff', 'NPO', 'RMU']

import urllib.request
def tofu(name):
    url = f'https://huggingface.co/datasets/locuslab/TOFU/resolve/main/{name}.json'
    return [json.loads(l) for l in urllib.request.urlopen(url).read().decode().splitlines() if l.strip()]
facts = tofu('forget10_perturbed')                      # 400 QA = 20 fictitious authors x 20
retain = tofu('retain90')
FEWSHOT = [(retain[i]['question'], retain[i]['answer']) for i in (0, 25, 50)]
author = np.arange(len(facts)) // 20
halves = author % 2                                      # half 0 / half 1, split by author
print(len(facts), 'facts;', len(set(author)), 'authors;', np.bincount(halves))
""")

md("## 4. Run all 10 probes on the 5 models (~10 min per model; resumes from Drive)")
code("""
SCORES = {}
for name, mid in MODELS.items():
    path = f'{OUT}/scores_{name}.npz'
    if os.path.exists(path):
        SCORES[name] = dict(np.load(path)); print('loaded', name); continue
    t = time.time()
    tok, model = P.load(mid)
    s, gens = P.run_probes(tok, model, facts, FEWSHOT)
    np.savez(path, **s); P.save_json(gens, f'{OUT}/gens_{name}.json')
    SCORES[name] = s
    P.free(model); print(name, f'{time.time()-t:.0f}s', {k: round(float(np.mean(v)), 3) for k, v in s.items()})
""")

md("""
## 5. Calibrate detections on the retain-only model (cross-fitted by author half)
Each probe's threshold is set so the retain-only model (which never saw these facts) is "detected" at most 5% of the
time on the *other* half of authors. The held-out false-positive rate is plugged into the model as f_j.

**Precondition:** the full model must detect at least 50% of facts with the direct probe; otherwise the prompt format
does not match how the checkpoints were trained, and the run is invalid (fix the format, do not reinterpret).
""")
code("""
THR, FPR = P.calibrate(SCORES['retain'], halves, max_fpr=0.05)
f = np.array([FPR[k] for k in P.PROBES])
DET = {name: P.detect(SCORES[name], THR) for name in MODELS}
print('false-positive rate per probe:', dict(zip(P.PROBES, f.round(3))))
for name, Y in DET.items():
    print(f'{name:9s} detection rate per probe:', dict(zip(P.PROBES, Y.mean(0).round(2))))
full_direct = DET['full'][:, 0].mean()
print('PRECONDITION full-model direct detection >= 0.50:', round(full_direct, 3), 'PASS' if full_direct >= 0.5 else 'FAIL')
""")

md("## 6. Fit occupancy models (primary Mh; M0 and Mh2 as sensitivity), bootstrap CIs, goodness of fit")
code("""
import occ
FIT = {}
for name, Y in DET.items():
    path = f'{OUT}/fit_{name}.json'
    if os.path.exists(path):
        FIT[name] = json.load(open(path)); print('loaded', name); continue
    res = {}
    for m in occ.MODELS:
        r = occ.fit(Y, f, m)
        lo, hi, _ = occ.bootstrap_ci(Y, f, m, B=300 if m == 'Mh' else 100)
        res[m] = {'psi': r['psi'], 'lo': lo, 'hi': hi, 'aic': r['aic'], 'p': r['p'].tolist(), 'extra': r['extra'].tolist()}
    res['gof_p_Mh'] = occ.gof_count_test(Y, occ.fit(Y, f, 'Mh'), f, B=100)
    res['naive'] = occ.naive_estimates(Y)
    FIT[name] = res; P.save_json(res, path)
    print(name, {m: (round(res[m]['psi'], 3), round(res[m]['lo'], 3), round(res[m]['hi'], 3)) for m in occ.MODELS},
          'GOF p', round(res['gof_p_Mh'], 3), res['naive'])
""")

md("""
## 7. Ground truth by relearning (Deeb & Roger 2024 style; ~8 short fine-tunes)
For each unlearned model and the retain-only control: fine-tune on the facts of one author half (3 epochs, lr 1e-5),
then test the *other* half with the direct, paraphrase and sampled16 probes. A fact counts as recovered if any of the
three detects it. Ground truth ψ* = recovery(unlearned) − recovery(retain-only after the same relearning), clipped to
[0, 1]. The subtraction removes facts that relearning could teach from scratch.
""")
code("""
REC = {}
for name in UNLEARNED + ['retain']:
    path = f'{OUT}/relearn_{name}.json'
    if os.path.exists(path):
        REC[name] = json.load(open(path)); print('loaded', name); continue
    rec = np.zeros(len(facts), int)
    for h in (0, 1):
        train = [facts[i] for i in np.where(halves != h)[0]]
        test_idx = np.where(halves == h)[0]
        tok, model = P.relearn(MODELS[name], train, epochs=3, lr=1e-5, bs=8, seed=h)
        s, _ = P.run_probes(tok, model, [facts[i] for i in test_idx], FEWSHOT, probes=['direct', 'paraphrase', 'sampled16'])
        d = np.stack([s[k] >= THR[k][test_idx] for k in ('direct', 'paraphrase', 'sampled16')], 1).any(1)
        rec[test_idx] = d.astype(int)
        P.free(model)
    REC[name] = {'recovered': rec.tolist(), 'rate': float(rec.mean())}
    P.save_json(REC[name], path); print(name, 'recovery rate', round(rec.mean(), 3))
TRUTH = {n: float(np.clip(REC[n]['rate'] - REC['retain']['rate'], 0, 1)) for n in UNLEARNED}
print('ground truth psi*:', TRUTH)
""")

md("""
## 8. Pre-registered gate (from PREREG.md)
Per unlearned checkpoint:
- (a) the 95% CI of ψ̂ (Mh) covers ψ*, or |ψ̂ − ψ*| ≤ 0.05;
- (b) |ψ̂ − ψ*| ≤ 0.5 × min(|single − ψ*|, |any-of-K − ψ*|);
- (c) ψ* ≥ 2 × single-probe rate (the usual metric undercounts by at least 2x).

**GO** if (a), (b), (c) hold on ≥ 2 of 3 checkpoints AND (d) ψ̂ on the retain-only model ≤ 0.05.
**NO-GO** if any-of-K is within 0.05 of ψ* on all 3 (no model needed), or GOF p < 0.05 on all 3, or (d) fails.
Otherwise: INCONCLUSIVE (write it up, decide with the author).
""")
code("""
rows, passed = [], 0
for n in UNLEARNED:
    r, t = FIT[n], TRUTH[n]
    psi, lo, hi = r['Mh']['psi'], r['Mh']['lo'], r['Mh']['hi']
    single, anyk = r['naive']['single_probe'], r['naive']['any_of_K']
    a = (lo <= t <= hi) or abs(psi - t) <= 0.05
    b = abs(psi - t) <= 0.5 * min(abs(single - t), abs(anyk - t))
    c = t >= 2 * single and t > 0.05
    passed += a and b and c
    rows.append(dict(model=n, truth=round(t, 3), psi=round(psi, 3), ci=(round(lo, 3), round(hi, 3)), single=round(single, 3),
                     any_of_K=round(anyk, 3), gof_p=round(r['gof_p_Mh'], 3), a=a, b=b, c=c))
d = FIT['retain']['Mh']['psi'] <= 0.05
anyk_close_all = all(abs(FIT[n]['naive']['any_of_K'] - TRUTH[n]) <= 0.05 for n in UNLEARNED)
gof_fail_all = all(FIT[n]['gof_p_Mh'] < 0.05 for n in UNLEARNED)
if anyk_close_all or gof_fail_all or not d:
    verdict = 'NO-GO'
elif passed >= 2:
    verdict = 'GO'
else:
    verdict = 'INCONCLUSIVE'
for row in rows: print(row)
print('retain-only psi-hat:', round(FIT['retain']['Mh']['psi'], 3), '(d)', d)
print('full-model psi-hat (sanity, should be high):', round(FIT['full']['Mh']['psi'], 3))
print('VERDICT:', verdict)
P.save_json({'rows': rows, 'd': d, 'anyk_close_all': anyk_close_all, 'gof_fail_all': gof_fail_all,
             'verdict': verdict, 'truth': TRUTH, 'fpr': FPR, 'full_direct': float(full_direct)}, f'{OUT}/GATE.json')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"]
    c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
(HERE / "kill_experiment.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote kill_experiment.ipynb with", len(cells), "cells")
