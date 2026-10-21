# Paper outline

A skeleton, not a draft. Each section says what it will cover. As the build goes, drop notes, numbers,
failures and the reasons behind decisions under the matching section, so the writing week starts from a
full notebook. Target: a 4–8 page workshop paper, with an appendix.

## Abstract

Written last. The question, what we did, the headline result, and why it matters.

## 1. Introduction

- **The problem:** systems that tip suddenly, and why early warning matters.
- **What existing warning statistics do,** and where they fall short.
- **Our question:** do warnings read from a learned world model come earlier or more reliably than classic ones?
- **Contributions,** as two or three bullets.
- **What we do not claim,** for example being first at latent-space warnings.

## 2. Background and related work

- Critical slowing down and classic early-warning signals.
- Machine learning that finds phase transitions in physics models.
- JEPA and self-supervised world models.
- Prior work on warnings in learned representations, and how ours differs.
- Early warnings in financial markets.

## 3. Method

- The world model: what it is, why one per domain, and how collapse is prevented.
- The warning signals we read from it.
- The baselines it must beat.
- The scoring protocol: what was fixed in advance, the surrogate tests, ROC, lead time, false alarms.
- The key design decisions and why we made them.

## 4. Experiment 1: Ising, "did it learn the physics?"

- The setup.
- The results: do the warnings peak at the critical point, and do they scale with system size?
- What the model's internals encode (the probes and their controls).
- Ablations.
- What failed or surprised us.

## 5. Experiment 2: Markets

- The data and the events (regime shifts, liquidation cascades).
- Warning results against the classic signals, across assets.
- The backtest against plain volatility targeting.
- What failed and why.

## 6. Triage benchmark: Jev

May move to the appendix, depending on page count.

- Does a decision model add anything over a threshold rule?
- How repeatable is it, how strong is its option-order bias, and does it judge better from words or from numbers?

## 7. System

Short; may move to the appendix.

- How the live pipeline runs end to end: the engine, the triage, the agents and the app.
- Enough to support the claim that the method works live, not just offline.

## 8. Discussion and limitations

- When the learned signals help, and when they don't.
- Honest limits: two domains, major coins only, live-period length.
- Future work: one shared model, and other domains such as seizures.

## 9. Conclusion

## Appendix

- Hyperparameters, the number of configurations tried, extra figures.
- Reproducibility: code, data sources and seeds.
