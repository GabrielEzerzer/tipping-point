# Paper outline

A skeleton, not a draft. Each section says what it will cover. As the build goes, drop notes, numbers,
failures and the reasons behind decisions under the matching section, so the writing week starts from a
full notebook. Target: a 4–8 page workshop paper, with an appendix.

## Abstract

Written last. The question, what we did, the headline result, and why it matters.

## 1. Introduction

- **The problem:** systems that tip suddenly, and why early warning matters.
- **The gap:** learned early-warning signals are claimed on synthetic data without ground truth; nothing forces them to be right.
- **Our claim:** finite-size scaling on Ising, where Tc and the exponents are exact, is a falsification test any learned early-warning signal must pass.
- **Contributions,** as two or three bullets: the test, its result on JEPA latent signals and on the classic indicators, and the same pre-registered protocol applied to markets.
- **What we do not claim:** being first at latent-space warnings (Fang & Li 2025 and others); that latent warnings beat classic indicators on Ising (that is the question, not the premise); anything about the agent layer or the app.

## 2. Background and related work

- Critical slowing down and classic early-warning signals.
- Machine learning that finds phase transitions in physics models.
- JEPA and self-supervised world models.
- Prior work on warnings in learned representations, and how ours differs.
- Early warnings in financial markets.

## 3. Method

- The world model: what it is, why one per domain, and how collapse is prevented.
- The warning signals we read from it: prediction-error surprise, latent variance, latent AC1.
- The falsification test: finite-size scaling as ground truth, plus linear probes for magnetisation and susceptibility.
- The classic raw-signal indicators, tested under the same protocol.
- The scoring protocol: what was fixed in advance, the surrogate tests, ROC, lead time, false alarms.
- The key design decisions and why we made them.

## 4. Experiment 1: Ising, the falsification test

- The setup: all-temperature training is primary; the held-out Tc band is the ablation.
- The results: does the latent signal obey finite-size scaling? Which signals peak at the critical point, and which are monotone in T?
- What the model's internals encode (the probes and their controls).
- Ablations.
- What failed or surprised us.

## 5. Experiment 2: Markets

- The protocol applied to markets; reported either way.
- The data and the events (regime shifts, liquidation cascades).
- Warning results against the classic signals, across assets.
- The backtest against plain volatility targeting.
- What failed and why.

## 6. Triage benchmark: Jev

Included only if Jev beats a threshold rule at #32; otherwise it stays an app feature and leaves the paper. May move to the appendix, depending on page count.

- Does a decision model add anything over a threshold rule?
- How repeatable is it, how strong is its option-order bias, and does it judge better from words or from numbers?

## 7. System

Short; may move to the appendix.

- How the live pipeline runs end to end: the engine, the triage, the agents and the app.
- Enough to support the claim that the method works live, not just offline.

## 8. Discussion and limitations

- When the learned signals help, and when they don't.
- What a fail would mean for the field: learned early-warning claims made without ground truth.
- Honest limits: two domains, major coins only, live-period length.
- Future work: one shared model, and other domains such as seizures.

## 9. Conclusion

## Appendix

- Hyperparameters, the number of configurations tried, extra figures.
- Reproducibility: code, data sources and seeds.
