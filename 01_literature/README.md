# Literature

Every arXiv entry below was checked against the arXiv API (ID, title, first author) on 2026-10-05; DOIs were checked
on Crossref. PDFs are not redistributed (licences); follow the links. Novelty-check reports are in `novelty_checks/`.

## Closest prior work

| Paper | PDF | What it does | Difference from us |
|---|---|---|---|
| Deeb & Roger, "Do Unlearning Methods Remove Information from Language Model Weights?" (2024) [2410.08827](https://arxiv.org/abs/2410.08827). **Status:** arXiv preprint only; submitted to ICLR 2025 and rejected (OpenReview `uDjuCpQH5N`). **78 citations** on Google Scholar (checked 2026-10-05) | yes | Fine-tune on some forgotten facts, test recovery of the others; ~88% of pre-unlearning accuracy recovered. Asks "removed or hidden?" directly | An attack needing fine-tuning access; no estimate of the share still stored, no presence/detection split. **We use it as ground truth.** Not found by the three novelty checks; found while designing the experiment |
| Scholten et al., "A Probabilistic Perspective on Unlearning and Alignment for LLMs" (ICLR 2025) [2410.03523](https://arxiv.org/abs/2410.03523) | yes | Greedy evaluation overstates unlearning; bounds on leakage from the sampled output distribution of a prompt | Sampling randomness for one prompt; no heterogeneous probes, no ψ vs p |
| Reisizadeh et al., "Leak@k" (2025) [2511.04934](https://arxiv.org/abs/2511.04934) | yes | Probability that at least one of k samples leaks | Empirical leak metric; no latent-variable model |
| Rybak et al., "REBEL" (2026) [2602.06248](https://arxiv.org/abs/2602.06248) | yes | Evolutionary prompt search recovers "forgotten" knowledge | Attack success rate, no estimator |
| Tran To et al., "Harry Potter is Still Here!" (LURK, EMNLP Findings 2025) [2505.17160](https://arxiv.org/abs/2505.17160) | yes | GCG-style suffixes surface unlearned facts | Attack, not estimator |
| Goel et al., "Auditing LM Unlearning via Information Decomposition" (2026) [2601.15111](https://arxiv.org/abs/2601.15111) | yes | Partial information decomposition of representations | White-box; no probe-based estimate |
| Lynch et al., "Eight Methods to Evaluate Robust Unlearning in LLMs" (2024) [2402.16835](https://arxiv.org/abs/2402.16835) | yes | Battery of robustness evaluations (paraphrase, relearning, ...) | Many probes, but reported separately; no joint model |
| Hu et al., "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning" (2024) [2406.13356](https://arxiv.org/abs/2406.13356) | yes | Small benign relearning brings knowledge back | Attack evidence for "hidden, not removed" |

## Ecology / statistics (the transferred tool)

| Paper | Link | Use |
|---|---|---|
| MacKenzie et al., "Estimating site occupancy rates when detection probabilities are less than one", Ecology 2002 | [doi:10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2](https://doi.org/10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2) | Base occupancy model (ψ, p) |
| Royle & Nichols, "Estimating abundance from repeated presence–absence data or point counts", Ecology 2003 | [doi:10.1890/0012-9658(2003)084[0777:EAFRPA]2.0.CO;2](https://doi.org/10.1890/0012-9658(2003)084[0777:EAFRPA]2.0.CO;2) | Heterogeneous detection |
| Royle & Link, "Generalized site occupancy models allowing for false positive and false negative errors", Ecology 2006 | [doi:10.1890/0012-9658(2006)87[835:GSOMAF]2.0.CO;2](https://doi.org/10.1890/0012-9658(2006)87[835:GSOMAF]2.0.CO;2) | False positives (our f_j) |
| Link, "Nonidentifiability of population size from capture–recapture data with heterogeneous detection probabilities", Biometrics 2003 | [doi:10.1111/j.0006-341x.2003.00129.x](https://doi.org/10.1111/j.0006-341x.2003.00129.x) | Main theory risk: ψ weakly identified under heterogeneity; why we report 3 models |

Journal PDFs are not stored (publisher copyright); open via DOI.

## Benchmarks and unlearning methods used

| Paper | PDF | Use |
|---|---|---|
| Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs" (2024) [2401.06121](https://arxiv.org/abs/2401.06121) | yes | Dataset (fictitious authors: the base model cannot know them) |
| Dorna et al., "OpenUnlearning" (2025) [2506.12618](https://arxiv.org/abs/2506.12618) | yes | Public checkpoints we audit |
| Zhang et al., "Negative Preference Optimization" (2024) [2404.05868](https://arxiv.org/abs/2404.05868) | yes | NPO method |
| Li et al., "The WMDP Benchmark" (2024) [2403.03218](https://arxiv.org/abs/2403.03218) | yes | RMU method |
| Shi et al., "MUSE" (ICLR 2025) [2407.06460](https://arxiv.org/abs/2407.06460), 361 citations on Google Scholar (2026-10-05) | yes | Real-entity unlearning benchmark (not used in v1) |

## Related ecology-in-LLM work (not unlearning)

| Paper | PDF | Note |
|---|---|---|
| Li et al., "Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know?" (KnowSum, 2025) [2506.02058](https://arxiv.org/abs/2506.02058) | yes | Unseen-species estimate of total knowledge |
| Xin et al., "UCS: Estimating Unseen Coverage for Improved In-Context Learning" (2026) [2604.12015](https://arxiv.org/abs/2604.12015) | yes | Good–Turing coverage for demonstrations |
| Liu et al., "Counting Species of Ideas: A Bayesian Capture–Recapture Ecology Framework for Estimating LLM Novelty", HICSS 2026 | [doi:10.24251/HICSS.2026.017](https://doi.org/10.24251/HICSS.2026.017) | Capture–recapture for idea diversity |
