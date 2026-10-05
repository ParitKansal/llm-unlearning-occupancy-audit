# Deeb & Roger, ICLR 2025 reviews (submission 3850, OpenReview `uDjuCpQH5N`) and lessons for this project

Source: OpenReview forum page, read on 2026-10-05. Decision: **Reject** (22 Jan 2025).
Final ratings: 5 (wshX, raised from lower), 3 (YJoy, confidence 5), 8 (STkt), 6 (qho8, confidence 5).

## What the reviewers said

**Meta-review (AC qydR):** finetuning on a few unlearned facts recovering knowledge "had been demonstrated by prior
work"; reviewers questioned novelty; "not fundamentally different from existing re-learning experiments"; the authors
said relearning lacks a clear baseline, "but it is unclear how the new approach fixes this."

**YJoy (3, reject):**
- Limited novelty: relearning as an unlearning metric already exists (WMDP/Li et al. 2024, Tarun et al. 2023, Lynch et al. 2024, Rosati et al. 2024, Henderson et al. 2023).
- No experiment comparing RTT with existing relearning-time metrics; claims like "sets a higher bar" are not substantiated; "too much speculation".
- To pass: either much broader evaluation (methods / metrics / datasets) or a proper theoretical justification of what the metric measures.
- Suggests "better suited for a workshop".

**qho8 (6):**
- Overlooks Lynch et al. 2024 (2402.16835) and Hu et al. 2024 (2406.13356).
- T and V come from the same distribution and were unlearned together, so they may share the same "hiding" mechanism; is that different from fine-tuning on part of the forget set?
- Asks for: held-out questions never unlearned; ablations on the size of T; fine-tuning on unrelated data; a fixed fine-tuning set the community can reuse as a gold standard.

**STkt (8):** unclear threat model / real-world use; "If RTT does not work, other attacks might still work" — failing to recover does not prove removal.

**wshX (5):** more unlearning methods (TOFU baselines), more formats than MCQ, clearer setup.

**Authors' useful findings:** relearning time and small-sample relearning have no baseline (recovery could be re-teaching);
independent facts (random birthdays) let them fine-tune as much as they want; for MCQ, loss on question + letter + answer
recovered the most hidden knowledge.

## Lessons for an occupancy-based audit

1. **Do not pitch "unlearning hides knowledge".** That is known (Lynch, Hu, WMDP, Deeb & Roger). The new part is an *estimator*: how much is still stored, with a CI, from black-box probes, plus how many probes an audit needs.
2. **Compare against existing metrics in experiments** (YJoy's main complaint): single probe, any-of-K, leak@k-style sampling, MCQ, relearning recovery. The kill gate requires beating the first two; a full study would add the rest.
3. **Bring theory** (YJoy's alternative path): identifiability (the detectability condition we found in simulation), bias of single-probe and any-of-K estimators, probes-needed bound.
4. **Have a baseline for relearning** (the AC's point): our retain-only model under the same relearning is exactly that baseline. 
5. **Relearning is a lower bound on what is stored** (STkt): treat ψ\* as a lower bound and say so; use the strongest relearning we can.
6. **Breadth** (YJoy, wshX): a full study needs ≥ 5 unlearning methods, 2 model sizes, TOFU + WMDP + MUSE, open-ended and MCQ formats.
7. **Ablate the relearning set** (qho8): size of T and an unrelated-data fine-tune.
8. **Threat model / use case** (STkt): a black-box auditor (regulator, data subject, third-party evaluator) who can query the model but not fine-tune it, and needs a number with error bars.

## Changes made to the kill experiment because of this (2026-10-05, before any GPU run)
- Relearning loss now covers the user question and the answer (authors found loss on question + answer recovers the most), not the answer only.
- A fact counts as recovered after relearning if **any of the 10 probes** detects it (was 3), so the ground truth is as strong as we can make it cheaply.
- PREREG states that ψ\* is a lower bound on stored knowledge.
