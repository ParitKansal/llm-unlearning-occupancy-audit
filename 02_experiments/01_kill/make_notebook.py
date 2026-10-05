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

**How to run fast (parallel):** open this notebook in 4 Colab A100 sessions. In section 3 set `RUN` differently in each:
`['full', 'retain']`, `['GradDiff']`, `['NPO']`, `['RMU']`, then Run all. Each session takes ~25–40 min and stops after
its GPU work. When all 4 are done, open one more fresh session (CPU is enough) with `RUN = []` and Run all: it does
the analysis and prints the verdict. Single-session alternative: `RUN = list(MODELS)` (~2.5 h, then analysis runs).
Every stage saves to Google Drive and is skipped on rerun (resume-safe). Public OpenUnlearning checkpoints are used, so
no unlearning training is needed.
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

# >>> set per session: ['full', 'retain'] | ['GradDiff'] | ['NPO'] | ['RMU'] | [] (analysis only) | list(MODELS) (all)
RUN = ['full', 'retain']

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

md("""
## 4. GPU work for the models in `RUN`: 10 probes, then relearning (resumes from Drive)
Relearning (Deeb & Roger 2024 style) for each unlearned model and the retain-only control: fine-tune on the facts of one
author half (3 epochs, lr 1e-5, loss on question and answer tokens), then run all 10 probes on the *other* half. Raw
scores are saved; detections are computed in the analysis with the retain-calibrated thresholds.
""")
code("""
for name in RUN:
    path = f'{OUT}/scores_{name}.npz'
    if os.path.exists(path):
        print('probes done:', name)
    else:
        t = time.time()
        tok, model = P.load(MODELS[name])
        s, gens = P.run_probes(tok, model, facts, FEWSHOT)
        np.savez(path, **s); P.save_json(gens, f'{OUT}/gens_{name}.json')
        P.free(model); print('probes', name, f'{time.time()-t:.0f}s', {k: round(float(np.mean(v)), 3) for k, v in s.items()})
    if name == 'full':   # early look at the precondition (formal check, with calibrated thresholds, is in section 5)
        d = np.load(path)['direct']
        print(f'EARLY CHECK full model: share of facts with direct-probe score >= 0.5 = {np.mean(d >= 0.5):.3f} '
              '(should be well above 0.5; if far below, stop and report)')
    if name not in UNLEARNED + ['retain']:
        continue
    rpath = f'{OUT}/relearn_scores_{name}.npz'
    if os.path.exists(rpath):
        print('relearning done:', name); continue
    t = time.time()
    rs = {k: np.full(len(facts), np.nan) for k in P.PROBES}
    for h in (0, 1):
        train = [facts[i] for i in np.where(halves != h)[0]]
        test_idx = np.where(halves == h)[0]
        tok, model = P.relearn(MODELS[name], train, epochs=3, lr=1e-5, bs=8, seed=h)
        s, _ = P.run_probes(tok, model, [facts[i] for i in test_idx], FEWSHOT)
        for k in P.PROBES:
            rs[k][test_idx] = s[k]
        P.free(model)
    np.savez(rpath, **rs); print('relearning', name, f'{time.time()-t:.0f}s')
print('GPU work finished for', RUN)
if RUN and set(RUN) != set(MODELS):      # worker session: make sure files reach Drive before the run stops
    drive.flush_and_unmount(); print('Drive flushed. This worker is done; the next cell stopping is expected.')
""")

md("""
## Analysis (runs only when all GPU work is on Drive)
In a worker session this cell stops the run with a message; that is expected.
""")
code("""
need = [f'scores_{n}.npz' for n in MODELS] + [f'relearn_scores_{n}.npz' for n in UNLEARNED + ['retain']]
missing = [x for x in need if not os.path.exists(f'{OUT}/{x}')]
assert not missing, f'Not ready for analysis (fine in a worker session). Missing on Drive: {missing}'
SCORES = {n: dict(np.load(f'{OUT}/scores_{n}.npz')) for n in MODELS}
RSCORES = {n: dict(np.load(f'{OUT}/relearn_scores_{n}.npz')) for n in UNLEARNED + ['retain']}
print('all inputs present')
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
        lo, hi, _ = occ.bootstrap_ci(Y, f, m, B=300 if m == 'Mh' else 100, n_jobs=-1)
        res[m] = {'psi': r['psi'], 'lo': lo, 'hi': hi, 'aic': r['aic'], 'p': r['p'].tolist(), 'extra': r['extra'].tolist()}
    res['gof_p_Mh'] = occ.gof_count_test(Y, occ.fit(Y, f, 'Mh'), f, B=100, n_jobs=-1)
    res['naive'] = occ.naive_estimates(Y)
    FIT[name] = res; P.save_json(res, path)
    print(name, {m: (round(res[m]['psi'], 3), round(res[m]['lo'], 3), round(res[m]['hi'], 3)) for m in occ.MODELS},
          'GOF p', round(res['gof_p_Mh'], 3), res['naive'])
""")

md("""
## 7. Ground truth from relearning
A fact counts as recovered after relearning if any of the 10 probes detects it (same thresholds as above).
ψ* = recovery(unlearned) − recovery(retain-only after the same relearning), clipped to [0, 1]. The subtraction removes
facts that relearning could teach from scratch. ψ* is a lower bound on what is stored (failed recovery ≠ removal).
""")
code("""
REC = {n: P.detect(RSCORES[n], THR).max(1) for n in RSCORES}
for n, r in REC.items():
    print(f'{n:9s} recovery rate after relearning: {r.mean():.3f}')
TRUTH = {n: float(np.clip(REC[n].mean() - REC['retain'].mean(), 0, 1)) for n in UNLEARNED}
print('ground truth psi*:', TRUTH)
P.save_json({'recovery': {n: float(r.mean()) for n, r in REC.items()}, 'truth': TRUTH}, f'{OUT}/relearn_summary.json')
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

def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def save(name, cs, gpu=True):
    nb = {"cells": [dict(c) for c in cs], "metadata": {"colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    if gpu:
        nb["metadata"].update({"accelerator": "GPU"})
        nb["metadata"]["colab"]["gpuType"] = "A100"
    for c in nb["cells"]:
        t = src(c)
        c["source"] = [l + "\n" for l in t.split("\n")[:-1]] + [t.split("\n")[-1]]
    (HERE / name).write_text(json.dumps(nb, indent=1))
    print("wrote", name, "with", len(cs), "cells")


def find(prefix):
    return next(k for k, c in enumerate(cells) if src(c).lstrip().startswith(prefix))


# Single-session notebook (everything, RUN = all models)
single = [dict(c) for c in cells]
k_cfg = find("import numpy as np, torch, probes as P")
single[k_cfg] = dict(single[k_cfg], source=src(single[k_cfg]).replace("RUN = ['full', 'retain']", "RUN = list(MODELS)"))
save("kill_experiment.ipynb", single)

# Shared pieces
k_setup = find("!pip")
k_code_md, k_occ, k_probes = find("## 1. Code"), find("%%writefile occ.py"), find("%%writefile probes.py")
k_gpu_md, k_gpu = find("## 4. GPU work"), find("for name in RUN:")
k_an = find("## Analysis")
base = [cells[k_setup], cells[k_code_md], cells[k_occ], cells[k_probes]]

WORKERS = [("1_worker_full_retain", ["full", "retain"], "probes on the full and retain-only models, relearning on retain-only (~35-40 min)"),
           ("2_worker_GradDiff", ["GradDiff"], "probes and relearning on the GradDiff-unlearned model (~25-30 min)"),
           ("3_worker_NPO", ["NPO"], "probes and relearning on the NPO-unlearned model (~25-30 min)"),
           ("4_worker_RMU", ["RMU"], "probes and relearning on the RMU-unlearned model (~25-30 min)")]
for name, run, what in WORKERS:
    title = {"cell_type": "markdown", "metadata": {}, "source": f"""# Kill experiment, worker {name[0]} of 4: {', '.join(run)}

This notebook does: {what}. Runtime: **A100 GPU**. Then **Run all**.

Run the 4 worker notebooks at the same time in 4 Colab sessions. When all 4 print **WORKER DONE**, open
`5_analysis.ipynb` in a fresh session (CPU is enough) and Run all to get the verdict. Rerunning resumes from Drive."""}
    cfg = dict(cells[k_cfg], source=src(cells[k_cfg]).replace("RUN = ['full', 'retain']", f"RUN = {run!r}"))
    done = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": f"print('WORKER DONE: {name}. Results are on Drive in', OUT)"}
    save(f"{name}.ipynb", [title] + base + [cells[k_cfg - 1], cfg, cells[k_gpu_md], cells[k_gpu], done])

title = {"cell_type": "markdown", "metadata": {}, "source": """# Kill experiment, step 5: analysis and verdict

Run this **after all 4 worker notebooks print WORKER DONE**. A CPU runtime is enough (~5-10 min). Run all.
It calibrates detections on the retain-only model, fits the occupancy models, computes the relearning ground truth, and
applies the pre-registered gate (`PREREG.md`). The last line is the verdict."""}
cfg = dict(cells[k_cfg], source=src(cells[k_cfg]).replace("RUN = ['full', 'retain']", "RUN = []"))
save("5_analysis.ipynb", [title] + base + [cells[k_cfg - 1], cfg] + cells[k_an:], gpu=False)
