# Is It Really Forgotten? Occupancy Models for Auditing LLM Unlearning

*Can a method that ecologists use to count hidden animals tell us how much an AI model still remembers after it was told to forget? This is the story of testing that idea carefully, and of what happened when it partly worked and partly failed.*

Parit Kansal · October 2026 · [paritkansal.in](https://paritkansal.in) · [OpenReview](https://openreview.net/profile?id=~Parit_Kansal1) · [GitHub](https://github.com/ParitKansal)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/results-dark.png">
  <img alt="Chart: for each forgetting method, the true share of facts still stored (black bar), our estimate with its likely range (blue), the usual one-question test (orange square) and 'any of 10 questions' (green triangle)." src="figures/results.png">
</picture>

*How to read the chart: the black bar is the truth. The blue dot is our estimate and the blue line its likely range. The orange square is what the usual test reports, and the green triangle counts a fact as known if any one of ten questions found it. The closer a mark sits to the black bar, the better that method is.*

---

## Why this question matters

Large language models learn an enormous amount from the text they are trained on, and some of it should not stay in them. A person may ask a company to delete their private details. A publisher may object to a copyrighted book being memorised. A safety team may want to remove step-by-step knowledge that could help someone cause harm. Training a large model again from the beginning, without the unwanted data, costs far too much, so researchers have invented shortcuts called **unlearning methods**. These methods adjust an existing model so that it "forgets" a chosen set of facts, which is usually called the *forget set*.

The obvious follow-up question is whether the forgetting actually worked, and this is where things become slippery. In most published work, a model is checked by asking it about each fact once. If it no longer gives the right answer, the fact is counted as forgotten, and the paper reports something like "95% of the forget set was removed". Several recent studies have shown that this is too generous. Ask the same thing in different words, ask it as multiple choice, let the model answer many times at random, or retrain it lightly on related material, and much of the "forgotten" knowledge comes back. The knowledge was hidden, not removed. What nobody had offered was a way to put a **number** on how much is still in there, with an honest margin of error.

## An idea borrowed from birdwatchers

Ecologists face almost exactly the same problem every day. Imagine walking through a forest looking for a rare bird. If you do not see it, you cannot conclude that it does not live there; it may simply have been hiding that morning. Ecologists solved this decades ago. They visit each site several times and record, visit by visit, whether the species was seen. A statistical tool called the **occupancy model** (MacKenzie and colleagues, 2002) then separates two things that a single visit mixes together: how likely it is that the species truly lives at a site, and how likely one visit is to spot it when it does. From the pattern of sightings and misses across many sites, it estimates how many sites are really occupied, including the ones where the bird was never seen.

The translation to AI models is direct. Each fact the model was told to forget plays the role of a forest. Each different way of asking about that fact plays the role of a visit. "The bird lives there" becomes "the model still stores the fact", and "the visit spotted the bird" becomes "this way of asking got the fact out of the model".

| Counting birds | Checking an AI model |
|---|---|
| a forest | one fact the model was told to forget |
| the bird lives there | the model still stores the fact |
| one visit to the forest | one way of asking about the fact |
| share of forests with the bird | share of "forgotten" facts the model still stores |
| chance one visit spots the bird | chance one way of asking reveals a stored fact |

A small example shows why this helps. Suppose we ask about three forgotten facts in four ways: a direct question, a reworded question, a multiple-choice question and a translated question. Fact A is revealed only by the multiple-choice question. Fact C is revealed by the reworded question and by multiple choice. Fact B is never revealed. The usual test only looks at the direct question, which failed for all three, so it declares all three forgotten. Yet A and C are clearly still stored. And because we can see that the direct question also missed A and C, we know it is a weak way of asking, so B might well be hidden too. Over hundreds of facts, the occupancy model turns exactly this kind of reasoning into a single estimate with a likely range, such as "the method claims 95% forgotten, but about 45%, give or take 6%, is still stored". A longer explanation is in [`docs/IDEA.md`](docs/IDEA.md).

## Planning the test before running it

New ideas are easy to fall in love with, so the test was planned to be able to fail. Before any experiment, three independent literature searches looked for anyone who had already used occupancy models, or anything like them, to check AI unlearning; none was found (the reports are in [`literature/novelty_checks/`](literature/novelty_checks/)). While designing the experiment, the closest related paper turned out to be Deeb and Roger (2024), who showed with a retraining attack that unlearned knowledge is often only hidden. Their paper was turned down at ICLR 2025, and reading the reviews carefully shaped this project: reviewers wanted a clear baseline, a comparison with existing measures, and honesty about what any test can and cannot prove. Those lessons are summarised in [`literature/deeb_roger_iclr2025_reviews.md`](literature/deeb_roger_iclr2025_reviews.md).

The most important decision was to write down the pass and fail rules **before** running anything, in a pre-registration ([`PREREG.md`](experiments/kill_v1/PREREG.md)). The method would pass only if its estimate landed close to the true amount of stored knowledge, beat the usual one-question test clearly, and stayed near zero on a model that had never learned the facts at all. Fixing these rules in advance means the result cannot be quietly reinterpreted afterwards to look better than it is.

Even before touching a real model, a computer simulation exposed a subtle trap. If a fact is "stored" but no way of asking ever reveals it more often than a lucky guess would, then it is mathematically impossible to tell it apart from a fact that is truly gone. In the simulation, this let the estimate drift up to 63% on data where nothing was stored. The fix was to define the quantity being measured as knowledge that is *detectably* stored, meaning that the questions must, on average, find it noticeably more often than chance. With that rule, the simulated estimates became accurate and the empty case stayed near zero. This was written into the pre-registration before the real run.

## The experiment

The test used TOFU, a public benchmark of 400 short facts about 20 made-up authors, such as where an invented novelist was born or what genre they write in. Because the authors are fictional, a model can only know these facts if it was trained on them, which makes the ground truth clean. Instead of training anything new, the test used five public versions of a small model, Llama-3.2-1B, released by the [OpenUnlearning](https://github.com/locuslab/open-unlearning) project. One had been trained on all the facts and had not forgotten anything. One had never been trained on these particular facts, which makes it the control: it should know nothing about them. The other three had been trained on the facts and then made to forget them using three popular unlearning methods called GradDiff, NPO and RMU.

Every one of the 400 facts was then asked about in ten different ways. These were a plain direct question, a reworded question, the question with an "expert" instruction, the question written as raw text without the chat format, the question after three worked examples, the question framed as a hypothetical encyclopedia entry, a request for the shortest possible answer, a fill-in-the-blank version with the start of the answer given, sixteen separate answers sampled at random, and a multiple-choice version where the model's preference among the true answer and five wrong ones was measured. An answer counted as revealing the fact only if it scored above a threshold chosen so that the control model, which cannot know the facts, crossed it on at most 5% of facts. The occupancy model was then fitted to the pattern of reveals and misses, and its likely range was computed by resampling the facts hundreds of times.

To know the real answer, each model was briefly retrained on the facts of half of the authors, and then checked on the other half. Retraining on some facts cannot teach a model the unrelated facts of other fictional people, so whatever comes back on the other half must have been stored all along. The same retraining was applied to the control model, and its recovery was subtracted, so that only genuinely stored knowledge counted. This is the method of Deeb and Roger, and it gives a lower bound on what is stored. The whole test ran on Google Colab in four parallel sessions and used about half an hour of a single A100 GPU.

## What happened

| Forgetting method | Truth | Our estimate (likely range) | Usual one-question test | Any of 10 questions |
|---|---|---|---|---|
| GradDiff | 61.5% | **65.4%** (58–75%) | 38.0% | 72.0% |
| NPO | 23.5% | **25.1%** (9–34%) | 7.5% | 37.0% |
| RMU | 64.7% | 32.0% (27–40%) | 19.3% | 53.0% |
| Control (never learned the facts) | 0% | 8.1% (2–12%) | 3.8% | 16.0% |

*Each number is the share of the 400 facts still stored. Full details are in [`RESULTS.md`](experiments/kill_v1/RESULTS.md), and the notebooks exactly as they ran, with every output, are in [`experiments/kill_v1/runs/2026-10-05/`](experiments/kill_v1/runs/2026-10-05/).*

The first finding confirmed why the question matters. For every forgetting method, retraining recovered far more than the usual one-question test suggested: 1.6 times more for GradDiff, 3.1 times more for NPO and 3.4 times more for RMU. A model that looks almost clean when asked once can still hold a large share of what it was supposed to forget. Even the model that had never forgotten anything was missed by the usual test on nearly a third of the facts it certainly knew, which shows how unreliable a single way of asking is.

The second finding was the encouraging one. For GradDiff and NPO, the occupancy estimate landed within four percentage points of the truth: 65.4% against 61.5%, and 25.1% against 23.5%. The usual test was off by 16 to 24 points on the same models. Where the knowledge could be reached by asking, the birdwatchers' method measured it well.

Then came the two failures. On the control model, which never learned the facts, the method estimated that 8.1% were stored, above the pre-registered limit of 5%. Looking closer showed why. Some answers can be guessed from the question itself, for example a profession or a genre that the question hints at. Every way of asking "finds" these facts, even in a model that never saw them, and the method read those lucky guesses as hidden memory. The second failure was more fundamental. The RMU method hides knowledge so thoroughly that none of the ten ways of asking could reach most of it, yet a little retraining brought 65% of the facts back. No method that only asks questions can count knowledge that no question can reach. For forgetting methods like RMU, which scramble the model's internal representations, a question-based audit can only ever give a minimum.

By the rules written in advance, the verdict is **NO-GO**: the method, in this form, does not pass. It is published here anyway, because the lessons are real and useful. The usual way of checking unlearning is badly optimistic. Repeated, varied questioning combined with an occupancy model can measure hidden knowledge accurately when that knowledge is reachable. And any future version has to handle guessable facts and has to be honest that some methods hide knowledge beyond the reach of questions. The design also shows what a careful, pre-registered test of a new evaluation idea can look like, including what to do when the answer is no.

## Details for researchers

The estimator is a single-season occupancy model with known false-positive rates per probe, calibrated on the never-trained model and cross-fitted by author half, with a normal random effect on facts for detection heterogeneity (two simpler and alternative models are reported as sensitivity checks). Confidence intervals come from 300 bootstrap resamples over facts, and goodness of fit from a parametric bootstrap on the distribution of detection counts. The estimand is *detectably stored* knowledge: the average detection probability of a stored fact must exceed the average false-positive rate by 0.10, which restores identifiability when little is stored. The ground truth is relearning recovery on held-out authors minus the control's recovery under identical relearning, which is a lower bound on stored knowledge. The main failure modes are false positives that cluster on guessable facts, which violate the conditional-independence assumption (the control's "any of 10" rate was 0.16 against about 0.30 expected under independence), and knowledge that is recoverable by fine-tuning but undetectable by any black-box probe.

The repository is organised as follows.

```
docs/IDEA.md                  plain explanation with a worked example
literature/                   related work, the ICLR 2025 reviews of the closest paper, novelty checks
experiments/kill_v1/
  PREREG.md                   rules written before the run
  RESULTS.md                  results and analysis
  src/                        estimator (occ.py), questions and retraining (probes.py), tests, notebook builder
  notebooks/                  Colab notebooks: four run in parallel, then one analysis
  runs/2026-10-05/            the notebooks exactly as run, with all outputs
figures/                      the chart and the script that draws it
```

To reproduce the result, open the four `*_worker_*` notebooks in [`experiments/kill_v1/notebooks/`](experiments/kill_v1/notebooks/) in separate Colab sessions with an A100 GPU and run them at the same time; each takes about 25 to 40 minutes. When all four have finished, run `5_analysis.ipynb` on a CPU runtime, and its last line prints the verdict. More detail is in [`experiments/kill_v1/README.md`](experiments/kill_v1/README.md). The estimator on its own can be checked against simulated data with `python experiments/kill_v1/src/test_occ.py`.

The closest related work is Deeb and Roger (2024), [*Do Unlearning Methods Remove Information from Language Model Weights?*](https://arxiv.org/abs/2410.08827), whose retraining test serves as the ground truth here. Scholten and colleagues (ICLR 2025), in [*A Probabilistic Perspective on Unlearning and Alignment for LLMs*](https://arxiv.org/abs/2410.03523), showed that judging a model by one greedy answer overstates forgetting. Reisizadeh and colleagues (2025), in [*Leak@k*](https://arxiv.org/abs/2511.04934), showed that asking many times brings forgotten facts back. The ecology method comes from MacKenzie and colleagues (2002), [*Estimating site occupancy rates when detection probabilities are less than one*](https://doi.org/10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2). A fuller annotated list is in [`literature/README.md`](literature/README.md). As of October 2026, no earlier use of occupancy models for checking AI unlearning was found.

## How to cite

If this idea or these results help your work, please cite this repository:

```bibtex
@misc{kansal2026occupancyunlearning,
  author       = {Kansal, Parit},
  title        = {Is It Really Forgotten? Occupancy Models for Auditing {LLM} Unlearning (a pre-registered negative result)},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/ParitKansal/llm-unlearning-occupancy-audit}},
  note         = {GitHub repository}
}
```

GitHub's **"Cite this repository"** button, on the right side of this page, gives the same reference in other formats.

## Notes and license

The literature searches and the code were developed with the help of an AI assistant (Claude); every experiment was run, and every result checked, by the author. The code is released under the [MIT License](LICENSE), and the text and figures under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
