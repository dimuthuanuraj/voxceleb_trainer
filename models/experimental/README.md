# `models/experimental/` — Quarantined model implementations

This directory holds **architectures that have been empirically shown
not to work** for speaker verification under this repository's training
pipeline. They are kept as research artifacts (so future contributors
can read the code that produced the negative result), not as
production-ready models.

## Why a separate directory?

The trainer loads models dynamically via
`importlib.import_module("models." + model_name)`. Files living
directly in `models/` are reachable through ordinary configs and might
be picked up by accident. By moving the failing implementations into
`models/experimental/`, configs that try to train them have to opt in
explicitly — e.g., `model: experimental.NestedSpeakerNet` in YAML.
This makes the production / experimental split visible at the config
level instead of buried in a comment.

## What lives here

| File | Status | Reference |
|---|---|---|
| `NestedSpeakerNet.py` | Three NaN-cascade failures across three stabilisation attempts; best stable variant is +88% worse EER than the baseline. **Do not use.** | [`research_logs/2025-12-29-nested-learning-experiment.md`](../../research_logs/2025-12-29-nested-learning-experiment.md), [`docs/bugfixes/BUGFIX-010-quarantine-nestedspeakernet.md`](../../docs/bugfixes/BUGFIX-010-quarantine-nestedspeakernet.md) |

## Promotion / demotion criteria

A model can be **promoted** out of `experimental/` and into the main
`models/` directory when **all four** of the following hold:

1. A documented training run reaches a published-baseline EER on at
   least one standard test set (VoxCeleb1-O, mini-VoxCeleb1, or the
   project's own SL test list).
2. The training is stable for ≥ 50 epochs without NaN, gradient
   explosion, or loss divergence.
3. A bugfix doc captures any architectural changes required to reach
   that stability, with reproducible config and seed.
4. The promotion is reviewed and approved by the repository owner.

Conversely, a model in `models/` should be **demoted** to
`experimental/` (or removed entirely) when:

1. Multiple independent stabilisation attempts have failed, **and**
2. The failure has been root-caused with empirical evidence (not just
   "didn't work in one run").

## What to do if you're considering reviving a quarantined model

1. **Read the linked research log carefully.** The log enumerates what
   was tried and why each attempt failed. Repeating a tried-and-failed
   approach with slightly different hyperparameters is the most common
   failure mode for revival attempts.
2. **Identify the root cause that was diagnosed.** For
   `NestedSpeakerNet` it was *gradient-path-count explosion combined
   with anti-correlated audio features*. Hyperparameter tuning cannot
   fix either of those — they require architectural changes.
3. **Design a follow-up experiment** that targets the root cause
   specifically (e.g., gradient-bypass routes, feature-decorrelation
   regularisation), not the symptoms.
4. **Open a new branch** for the revival work. Don't modify the
   quarantined file in `main` — its purpose is to document the
   negative result as it was at the time of quarantine.

## Why we don't just delete these files

Three reasons:

- **Negative results are valuable.** The code is empirical evidence of
  *why* an approach didn't work. Deleting it forces every future
  contributor who has the same architectural idea to re-discover the
  failure from scratch.
- **The research logs already cite the file paths.** Deleting would
  break those references and the documented experiment history would
  partially lose its citation graph.
- **The architectural diagram and visualisation script
  (`visualize_nested_architecture.py`, `nested_architecture_diagram.{pdf,png}`
  at the repo root) reference this code.** Quarantining preserves
  those tooling links.

If a file in this directory becomes truly irrelevant (e.g., its
research log gets superseded, no one cites it anymore, and the
architecture is broken in a way no one would reattempt), it can be
deleted in a separate cleanup pass — not as part of the quarantine
itself.
