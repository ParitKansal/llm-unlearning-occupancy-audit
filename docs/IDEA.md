# The idea in plain words

## Unlearning
An LLM learns many facts in training. Sometimes we must make it forget some of them (private data, copyrighted text,
dangerous knowledge). Retraining from zero is too expensive, so *unlearning methods* (GradDiff, NPO, RMU, ...) change
the model so it "forgets" a chosen **forget set**.

## How forgetting is checked today
Usually one question per fact. Wrong answer → "forgotten". Then: "95% of the forget set is forgotten."
But knowledge often comes back with a paraphrase, another format, sampling, or a little fine-tuning.
One failed question does not prove the fact is gone; the model may still **store** it and only **hide** it.

## The ecology tool
Ecologists who do not see a rare bird on one visit do not conclude it is absent. They visit each site several times
and fit an **occupancy model** (MacKenzie et al. 2002), which separates:
- **ψ** — probability the species is really there;
- **p** — probability one visit detects it when it is there.

| Ecology | Unlearning |
|---|---|
| site | one forgotten fact |
| species present | fact still stored in the model |
| visit | one probe type |
| detection | the probe gets the fact out |
| ψ | share of "forgotten" facts still stored |
| p | chance a probe type reveals a stored fact |

## Example
| Fact | Direct | Paraphrase | Multiple choice | Translation |
|---|---|---|---|---|
| A | 0 | 0 | 1 | 0 |
| B | 0 | 0 | 0 | 0 |
| C | 0 | 1 | 1 | 0 |

Usual metric (direct question only): 100% forgotten. A and C are clearly still stored. B was never detected, but the
probes are weak, so B may be hidden too; the model estimates how likely. Over thousands of facts the result is, e.g.,
"claimed 95% forgotten, but 45% ± 6% still stored", plus how many probe types an audit needs.

## Checking that the estimate is right
- **Relearning:** fine-tune the unlearned model on some forgotten facts; stored facts in the *other* part come back
  (Deeb & Roger 2024 use this to test removal).
- **Control:** a model that never saw the facts must give ψ ≈ 0.

## Why it seemed worth testing
No prior work found (three independent prior-work checks), it matters (unlearning claims back privacy, copyright and
safety), and it is cheap to test. In one line:
*"Ecologists never conclude a species is gone after one empty visit; unlearning evaluations do."*

## Main risks
Forgetting may be graded rather than yes/no; probes may fail together; relearning is an imperfect ground truth.
The kill experiment ([`experiments/kill_v1/`](../experiments/kill_v1/)) tested this; result: [NO-GO](../experiments/kill_v1/RESULTS.md).
