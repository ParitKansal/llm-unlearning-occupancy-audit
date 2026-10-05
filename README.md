# Occupancy models for unlearning audits

**Question.** After an LLM "unlearns" a set of facts, how many are still stored in its weights?
Today one failed probe counts as "forgotten". Ecology treats one empty visit as a non-detection, not an absence, and
uses occupancy models to separate presence (ψ) from detection (p). We bring that to unlearning: each fact is a site,
each probe type a visit, and the output is the share of the forget set still stored, with a confidence interval.

Status: **kill experiment ready, not run.**

| Folder | Contents |
|---|---|
| `00_background/` | `IDEA.md` (plain explanation) |
| `01_literature/` | `README.md` (annotated paper list), `novelty_checks/` (3 independent prior-work checks) |
| `02_experiments/01_kill/` | `PREREG.md` (go / no-go rule), `kill_experiment.ipynb` (Colab), `occ.py`, `probes.py`, `test_occ.py`, `make_notebook.py` |

## Run the kill experiment
1. Open `02_experiments/01_kill/kill_experiment.ipynb` in Colab (A100).
2. Run all. Results go to `MyDrive/occupancy_unlearning/kill_v1/`; reruns resume.
3. The last cell prints the verdict (GO / NO-GO / INCONCLUSIVE) and writes `GATE.json`.

Edit `occ.py` / `probes.py`, then `python make_notebook.py` to rebuild the notebook. `python test_occ.py` checks the
estimator on simulated data (CPU).

## Novelty status (2026-10-05)
NOT FOUND in three checks (arXiv + OpenReview; Semantic Scholar + web; Gemini Deep Research). Closest:
Deeb & Roger 2024 (relearning attack to test removal, our ground truth), Scholten et al. ICLR 2025, Leak@k.
Recheck before any follow-up work.
