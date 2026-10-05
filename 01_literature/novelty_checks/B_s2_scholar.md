# Novelty check B: Semantic Scholar + web (2026-10-05)

Source access:
- **Google Scholar: not reachable** (CAPTCHA), so web search was used instead (arXiv, ACL Anthology, LessWrong/Alignment Forum, ecology journals).
- **Semantic Scholar: partly reachable.** Keyword search was rate-limited (about 10 of 36 queries returned). The citation chain of MacKenzie et al. (2002) was scanned in full (4,652 citing papers), filtered for LLM/neural keywords.
- Every item below was opened and its title, authors and content checked.

Run with an AI research agent (Claude) and checked by the author. This copy keeps only the section on this repository's idea.

## Occupancy model x residual knowledge after unlearning: **NOT FOUND**
No paper applies occupancy or imperfect-detection models (estimating psi separately from p) to LLM knowledge or unlearning. Among the 4,652 papers citing MacKenzie 2002, every ML-related one is about ecology (camera traps, bioacoustics, occupancy from classifier outputs). None is about LLMs.

Closest items:
- https://arxiv.org/abs/2511.04934 | Leak@k: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding | Reisizadeh | 2025 (rev. 2026) | arXiv | Repeated sampling resurfaces "forgotten" knowledge; defines leak@k | Uses repeated sampling as an empirical leak metric, with no latent presence/detection model, no prevalence estimate with CIs, and no mix of probe types | verified: yes
- https://arxiv.org/abs/2506.02058 | Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know? (KnowSum) | Xiang Li | 2025 | arXiv | Estimates unseen knowledge by extrapolating from how often observed knowledge appears (unseen-species style) | Counts total knowledge, not presence vs. detection per fact; nothing on unlearning | verified: yes
- https://arxiv.org/abs/2504.14798 | Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models (RUB) | Hao Xuan | 2025/2026 | CVPR Workshops 2026 | Adversarial "unlearning mapping attack" to detect residual knowledge | Detection by attack, with no statistical model of detection probability; vision and diffusion models | verified: yes


## Queries
- C1: occupancy model LLM; unlearning repeated probes residual knowledge; imperfect detection knowledge probing; capture-recapture LM knowledge; hidden-not-removed relearning; occupancy+LLM detection probability; worst-case paraphrase probes unlearning; plus the MacKenzie 2002 citation chain.
