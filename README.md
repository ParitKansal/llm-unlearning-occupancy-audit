<div align="center">

# Is It Really Forgotten?
### Occupancy Models for Auditing Unlearning in Large Language Models

**Parit Kansal**
Independent Researcher · [paritkansal.in](https://paritkansal.in) · [OpenReview](https://openreview.net/profile?id=~Parit_Kansal1) · [GitHub](https://github.com/ParitKansal)

October 2026

[![Status: pre-registered negative result](https://img.shields.io/badge/status-pre--registered%20negative%20result-8a5cf6)](experiments/kill_v1/PREREG.md)
[![Verdict: NO-GO](https://img.shields.io/badge/verdict-NO--GO-e34948)](experiments/kill_v1/RESULTS.md)
[![Code: MIT](https://img.shields.io/badge/code-MIT-2a78d6)](LICENSE)
[![Text: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-1baf7a)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-52514e)](experiments/kill_v1/src/)
[![Runs on Colab](https://img.shields.io/badge/runs%20on-Google%20Colab-eda100)](experiments/kill_v1/notebooks/)

</div>

---

## Abstract

Unlearning methods are meant to remove specific knowledge from a trained language model, yet they are usually evaluated by asking about each fact once and counting a wrong answer as success. We ask whether a tool from ecology can measure what such tests miss. Ecologists estimate how many sites a species truly occupies from repeated, imperfect surveys using **occupancy models**, which separate the probability that a species is present from the probability that a survey detects it. We treat each fact in the forget set as a site and each of ten different ways of asking about it as a survey, and we estimate the share of "forgotten" facts that the model still stores. In a pre-registered test on the TOFU benchmark with public Llama-3.2-1B checkpoints unlearned by GradDiff, NPO and RMU, judged against a relearning ground truth, we find that the standard single-question test underestimates stored knowledge by a factor of 1.6 to 3.4. The occupancy estimate is accurate to within four percentage points for GradDiff and NPO. However, it reports 8.1% stored knowledge on a control model that never learned the facts, above the pre-registered limit of 5%, and it recovers only half of the knowledge hidden by RMU, which no question-based probe can reach. By the rules fixed in advance, the method fails in its current form. We release the design, code, executed notebooks and analysis so that the positive findings and the two failure modes can inform future unlearning audits.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/results-dark.png">
  <img width="900" alt="Dot plot. For GradDiff, NPO, RMU and a never-trained control: the relearning truth (black bar), the occupancy estimate with its 95% interval (blue), the single-question test (orange square) and the any-of-ten-questions rate (green triangle)." src="figures/results.png">
</picture>
</p>

**Figure 1.** Share of the forget set still stored, by unlearning method. The black bar is the ground truth from relearning; the blue dot and line are the occupancy estimate and its 95% interval; the orange square is the standard single-question test; the green triangle counts a fact as known if any of the ten questions revealed it. Marks closer to the black bar are more accurate.

**Contents:** [1. Motivation](#1-motivation) · [2. Approach](#2-approach) · [3. Study design](#3-study-design) · [4. Results](#4-results) · [5. Discussion](#5-discussion) · [6. Limitations](#6-limitations) · [7. Reproducibility](#7-reproducibility) · [8. Related work](#8-related-work) · [Citation](#citation)

---

## 1. Motivation

Large language models absorb a vast amount of information during training, and some of it should not remain in them: personal data that a person has asked to be deleted, copyrighted text, or knowledge that could help someone cause harm. Retraining a large model from scratch without that data is prohibitively expensive, so a family of shortcuts known as **unlearning methods** has been developed. These methods adjust an already-trained model so that it no longer reproduces a chosen set of facts, called the *forget set*.

Whether these methods succeed is usually judged in a simple way. Each fact is put to the model once, and if the model no longer answers correctly, the fact is counted as forgotten. Several studies have shown that this is too generous: rephrasing the question, offering multiple choices, sampling many answers, or lightly retraining the model often brings the "forgotten" knowledge back. The knowledge was hidden rather than removed. What has been missing is a way to state how much knowledge remains, as a number with an honest margin of error, using only the model's answers.

## 2. Approach

Field ecologists face the same difficulty when surveying wildlife. Failing to see a rare bird during one visit does not show that the bird is absent from a forest; it may simply have gone unseen. The standard remedy is to survey each site several times and analyse the record of sightings with an **occupancy model** (MacKenzie et al., 2002). The model separates two quantities that a single survey confounds: the probability that the species truly occupies a site, and the probability that a single survey detects it when it does. From the pattern of detections and non-detections across many sites, it estimates the true proportion of occupied sites, including those where the species was never observed.

The correspondence with unlearning is direct, as Table 1 shows. Each fact in the forget set takes the place of a site, and each distinct way of asking about the fact takes the place of a survey.

**Table 1.** Correspondence between wildlife surveys and unlearning audits.

| Wildlife survey | Unlearning audit |
|---|---|
| Site (for example, a forest) | One fact in the forget set |
| Species present at the site | Fact still stored in the model |
| One survey visit | One way of asking about the fact |
| Proportion of occupied sites | Proportion of "forgotten" facts still stored |
| Probability that a visit detects the species | Probability that a question reveals a stored fact |

A small example illustrates the benefit. Suppose three forgotten facts are each asked about in four ways: directly, reworded, as multiple choice, and in translation. The first fact is revealed only by the multiple-choice question; the second is revealed by both the reworded and the multiple-choice questions; the third is never revealed. A test based on the direct question alone declares all three forgotten. The pattern, however, shows that the first two are still stored, and because the direct question also missed those two, it is evidently a weak way of asking, so the third may be hidden as well. Across hundreds of facts, the occupancy model turns this reasoning into a single estimate with a confidence interval. A longer, non-technical explanation is given in [`docs/IDEA.md`](docs/IDEA.md).

## 3. Study design

**Prior work and pre-registration.** Before any experiment, three independent literature searches found no previous use of occupancy models, or comparable presence-and-detection models, for evaluating unlearning ([`literature/novelty_checks/`](literature/novelty_checks/)). The closest study, Deeb and Roger (2024), showed with a retraining attack that unlearned knowledge is frequently hidden rather than removed; its public reviews from ICLR 2025 informed this design and are summarised in [`literature/deeb_roger_iclr2025_reviews.md`](literature/deeb_roger_iclr2025_reviews.md). All pass and fail criteria were fixed in a pre-registration before any model was run ([`PREREG.md`](experiments/kill_v1/PREREG.md)). The method would pass only if its estimate matched the ground truth, clearly outperformed the standard test, and remained close to zero on a model that had never learned the facts.

**An identifiability condition, found in simulation.** A simulation study run before the experiment revealed that a fact which is "stored" but revealed no more often than a lucky guess cannot, even in principle, be distinguished from a fact that is absent. Without a safeguard, the estimate rose as high as 63% on simulated data in which nothing was stored. The quantity of interest was therefore defined as *detectably stored* knowledge: on average, the questions must reveal a stored fact noticeably more often than chance. With this definition, the simulated estimates were accurate and stayed near zero when nothing was stored. The definition was added to the pre-registration before the real experiment.

**Data and models.** The experiment uses TOFU, a public benchmark of 400 short facts about 20 fictitious authors. Because the authors are invented, a model can know these facts only if it was trained on them. Five public checkpoints of Llama-3.2-1B-Instruct from the [OpenUnlearning](https://github.com/locuslab/open-unlearning) project were evaluated: a model trained on all facts with no unlearning; a control model never trained on these facts; and three models trained on the facts and then unlearned with GradDiff, NPO and RMU.

**Probing.** Every fact was queried in ten ways: a direct question; a paraphrased question; the question with an expert instruction; the question as raw text outside the chat format; the question after three solved examples; the question framed as an encyclopedia entry; a request for the shortest possible answer; a fill-in-the-blank version with the start of the answer supplied; sixteen independently sampled answers; and a multiple-choice comparison of the true answer against five plausible wrong ones. A response counted as revealing the fact only if it exceeded a threshold set so that the control model, which cannot know the facts, crossed it on at most 5% of facts. Thresholds were calibrated on one half of the authors and applied to the other half.

**Estimation and ground truth.** The occupancy model was fitted to the record of reveals and misses, with 95% intervals obtained by resampling facts 300 times. The ground truth was measured independently: each model was briefly retrained on the facts of one half of the authors and then tested on the other half. Training on some fictitious people cannot teach a model unrelated facts about others, so whatever returns must have been stored. The same procedure was applied to the control model and its recovery was subtracted. This follows Deeb and Roger (2024) and yields a lower bound on stored knowledge. The full experiment ran on Google Colab in four parallel sessions and used about half an hour of a single A100 GPU.

## 4. Results

**Table 2.** Share of the 400 forget-set facts still stored, by method and estimator.

| Unlearning method | Ground truth | Occupancy estimate (95% interval) | Single-question test | Any of ten questions |
|---|---:|---:|---:|---:|
| GradDiff | 61.5% | **65.4%** (58.0–75.1) | 38.0% | 72.0% |
| NPO | 23.5% | **25.1%** (8.5–34.2) | 7.5% | 37.0% |
| RMU | 64.7% | 32.0% (26.7–39.6) | 19.3% | 53.0% |
| Control (never trained on the facts) | 0.0% | 8.1% (2.1–11.7) | 3.8% | 16.0% |

The relearning ground truth exceeded the single-question test for every method, by a factor of 1.6 for GradDiff, 3.1 for NPO and 3.4 for RMU. Even on the model that had never undergone unlearning, the direct question failed on nearly a third of facts that the model demonstrably knew. For GradDiff and NPO, the occupancy estimate fell within four percentage points of the ground truth, whereas the single-question test was off by 16 to 24 points. For RMU, the estimate recovered only about half of the stored knowledge. On the control model, the estimate was 8.1%, exceeding the pre-registered limit of 5%. The pre-registered verdict is therefore **NO-GO**. Complete results, including two alternative model specifications and goodness-of-fit tests, are reported in [`RESULTS.md`](experiments/kill_v1/RESULTS.md), and the notebooks exactly as executed are in [`experiments/kill_v1/runs/2026-10-05/`](experiments/kill_v1/runs/2026-10-05/).

## 5. Discussion

The results support the central premise that a single question is an unreliable measure of forgetting. Models that appear largely clean under the standard test still hold a substantial share of what they were meant to forget, and the gap is large enough to change conclusions about which method works. Where hidden knowledge can be reached by some form of questioning, as with GradDiff and NPO, combining varied questions with an occupancy model measured it accurately and far better than either the single-question test or a simple count of facts revealed by any question.

Two findings explain the failure. The first is that false positives are not independent. Some answers can be inferred from the question itself, such as a profession or genre that the wording suggests, and every way of asking tends to "find" these facts, even in a model that never saw them. On the control model, the share of facts revealed by at least one question was 16%, roughly half of what independent errors would produce, which shows that the errors cluster on the same facts. The model interprets this cluster as hidden knowledge. A future version would need to model such guessable facts explicitly, for example by using the control model's responses at the level of individual facts.

The second finding is more fundamental. RMU, which disrupts the model's internal representations, hides knowledge so thoroughly that none of the ten questions could reveal most of it, yet retraining recovered 65% of the facts. No method that relies only on the model's answers can count knowledge that no question can elicit. For methods of this kind, question-based audits can provide only a lower bound, and claims of removal require access to the model's weights or to fine-tuning.

## 6. Limitations

This study evaluates a single small model, a single benchmark of fictitious facts, and one public checkpoint for each of three unlearning methods, so the results may not extend to larger models, real-world knowledge, or other unlearning techniques. The ground truth is itself a lower bound, because a failed recovery does not prove that knowledge was removed, and it relies on the control model as a baseline. The ten questions are a reasonable but not exhaustive set, and the scoring of free-text answers by word overlap may misclassify some responses. These choices were appropriate for a low-cost, pre-registered test of feasibility, but a full study would need broader coverage.

## 7. Reproducibility

The repository contains everything needed to repeat the experiment.

```
docs/IDEA.md                  non-technical explanation with a worked example
literature/                   annotated related work, ICLR 2025 reviews of the closest study, novelty checks
experiments/kill_v1/
  PREREG.md                   pre-registered design and decision rule
  RESULTS.md                  full results and analysis
  src/                        occupancy estimator (occ.py), probes and relearning (probes.py), tests, notebook builder
  notebooks/                  Colab notebooks: four parallel workers and one analysis notebook
  runs/2026-10-05/            the executed notebooks, with all outputs
figures/                      Figure 1 and the script that produces it
```

To reproduce the results, open the four `*_worker_*` notebooks in [`experiments/kill_v1/notebooks/`](experiments/kill_v1/notebooks/) in separate Google Colab sessions with an A100 GPU and run them concurrently; each requires about 25 to 40 minutes. When all four have finished, run `5_analysis.ipynb` on a CPU runtime; its final line reports the verdict. Further instructions are given in [`experiments/kill_v1/README.md`](experiments/kill_v1/README.md). The estimator can be validated independently on simulated data with `python experiments/kill_v1/src/test_occ.py`.

**Technical summary.** The estimator is a single-season occupancy model with known, probe-specific false-positive rates (calibrated on the control model and cross-fitted by author) and a normal random effect on facts to capture heterogeneous detectability; a constant-detection model and a two-class mixture are reported as sensitivity analyses. Intervals use 300 nonparametric bootstrap resamples over facts, and goodness of fit uses a parametric bootstrap on the distribution of detection counts. Identifiability is enforced by requiring the mean detection probability of a stored fact to exceed the mean false-positive rate by 0.10.

## 8. Related work

Deeb and Roger (2024), [*Do Unlearning Methods Remove Information from Language Model Weights?*](https://arxiv.org/abs/2410.08827), introduced the retraining test used here as ground truth and showed that unlearned knowledge is often recoverable. Scholten et al. (ICLR 2025), [*A Probabilistic Perspective on Unlearning and Alignment for Large Language Models*](https://arxiv.org/abs/2410.03523), showed that evaluating a single greedy answer overstates forgetting. Reisizadeh et al. (2025), [*Leak@k: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding*](https://arxiv.org/abs/2511.04934), showed that repeated sampling resurfaces forgotten facts. The ecological method originates with MacKenzie et al. (2002), [*Estimating Site Occupancy Rates When Detection Probabilities Are Less Than One*](https://doi.org/10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2). An annotated bibliography is provided in [`literature/README.md`](literature/README.md). As of October 2026, we found no earlier application of occupancy models to the evaluation of unlearning.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{kansal2026occupancyunlearning,
  author       = {Kansal, Parit},
  title        = {Is It Really Forgotten? Occupancy Models for Auditing Unlearning in Large Language Models},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/ParitKansal/llm-unlearning-occupancy-audit}},
  note         = {Pre-registered negative result. GitHub repository}
}
```

The **Cite this repository** button on the right of this page provides the same reference in other formats.

## Acknowledgements

This work uses the TOFU benchmark and the public checkpoints released by the OpenUnlearning project. Literature searches and code were developed with the assistance of an AI system (Claude); all experiments were run and all results verified by the author.

## License

The code is released under the [MIT License](LICENSE). The text and figures are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
