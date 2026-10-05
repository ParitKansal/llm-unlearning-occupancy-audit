# Novelty check A: arXiv + OpenReview (2026-10-05)

Sources: arXiv API (abstract-field boolean queries); OpenReview api2 search (all venues, including ICLR 2027 submissions, NeurIPS/ICML 2025–26 workshops and TMLR); a keyword search over the NeurIPS 2025 and 2026 accepted-paper lists.
Verification: arXiv papers were opened on their arxiv.org/abs page (title, authors, date, abstract). OpenReview forum pages block automated access, so OpenReview-only papers were checked through the api2 search record (title, venue, authors when not anonymous, abstract), marked "verified: yes (OR API)".

Run with an AI research agent (Claude) and checked by the author. This copy keeps only the section on this repository's idea.

## Occupancy models (imperfect detection) x unlearning residual knowledgeNo paper uses occupancy / imperfect-detection models (psi vs p) to measure residual knowledge after unlearning. Related work either attacks harder to recover knowledge or uses probabilistic sampling metrics.
- 2410.03523 | A Probabilistic Perspective on Unlearning and Alignment for LLMs | Scholten | 2024 | ICLR 2025 Oral | high-probability bounds on the output distribution under sampling; shows greedy evaluation wrongly reports that unlearning succeeded | handles sampling randomness for one probe; no latent-presence vs detection split, no repeated diverse probes, no estimate of how many facts remain | verified: yes
- 2602.06248 | REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop | Rybak | 2026 | arXiv | evolutionary adversarial prompts recover "forgotten" TOFU/WMDP knowledge (attack success rate) | treats each probe as a pass/fail attack; no statistical model of detectability, no CIs | verified: yes
- 2505.17160 | Harry Potter is Still Here! (LURK) | To | 2025 | arXiv | adversarial suffixes surface latent unlearned knowledge | an attack, not an estimator | verified: yes
- 2601.15111 | Auditing Language Model Unlearning via Information Decomposition | Goel | 2026 | EACL 2026 | PID on representations to quantify residual knowledge | white-box representation measure; no repeated probes or occupancy model | verified: yes

Queries: occupancy model + language model; imperfect detection + unlearning; detection probability + unlearning; unlearning + residual knowledge; zero-inflated + language model; unlearning + paraphrase/multilingual evaluation; OR: "occupancy model unlearning", "imperfect detection knowledge", "unlearning evaluation latent presence detection probability"; CSV regex unlearn x (occupancy|detection probability|imperfect).
