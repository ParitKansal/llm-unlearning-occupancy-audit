# Kill experiment v1

Pre-registered test of occupancy models for unlearning audits on TOFU with Llama-3.2-1B (run 2026-10-05).

| File | Contents |
|---|---|
| [`PREREG.md`](PREREG.md) | Design and go / no-go rule, written before any GPU run |
| [`RESULTS.md`](RESULTS.md) | Results, verdict (NO-GO) and analysis |
| [`src/occ.py`](src/occ.py) | Occupancy models (M0, Mh, Mh2) with known false positives, bootstrap CIs, goodness of fit |
| [`src/probes.py`](src/probes.py) | The 10 probes, scoring, threshold calibration, relearning |
| [`src/test_occ.py`](src/test_occ.py) | Estimator check on simulated data (CPU) |
| [`src/make_notebook.py`](src/make_notebook.py) | Builds the notebooks (they embed `occ.py` and `probes.py`, so they run without this repo) |
| [`notebooks/`](notebooks/) | Clean Colab notebooks |
| [`runs/2026-10-05/`](runs/2026-10-05/) | The executed notebooks, with outputs |

## Run
1. Open `notebooks/1_worker_full_retain.ipynb`, `2_worker_GradDiff.ipynb`, `3_worker_NPO.ipynb` and `4_worker_RMU.ipynb` in four Colab sessions with an A100 GPU, and Run all in each. They save scores to `MyDrive/occupancy_unlearning/kill_v1/` and finish in about 25–40 minutes.
2. When all four print `WORKER DONE`, open `notebooks/5_analysis.ipynb` (CPU runtime; more cores is faster) and Run all. The last line is the verdict.

Single-session alternative: `notebooks/kill_experiment.ipynb` runs everything in one go (about 2 hours).
Every stage saves to Google Drive and is skipped on rerun.

## Rebuild the notebooks
Edit `src/occ.py` or `src/probes.py`, then `python src/make_notebook.py`.
