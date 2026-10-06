#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mathematical specification of every loss function in ``loss/``.

Each entry is recorded verbatim into an experiment's ``manifest.json`` so that
the analysis can reason about *why* a configuration behaved as it did, without
re-reading the loss source.  The fields are deliberately machine-readable:

    family          classification | metric | hybrid
    objective       LaTeX for the per-batch loss
    geometry        what the loss does to the embedding space
    hyperparams     which CLI flags actually reach this loss
    n_classes_dep   True if the loss instantiates a weight matrix of size
                    (nOut x nClasses) -- these scale with speaker count and are
                    the ones whose behaviour changes between the si-only,
                    ta-only and combined conditions
    batch_dep       True if the loss reads structure across the batch, i.e.
                    needs --nPerSpeaker > 1
    predicted_effect  the a-priori hypothesis this study tests

The `predicted_effect` fields are written BEFORE any run, so the analysis can be
scored against a pre-registered prediction rather than rationalised after the
fact.
"""

from __future__ import annotations

LOSS_SPECS = {
    # -------------------------------------------------------------- softmax
    "softmax": {
        "family": "classification",
        "display": "Vanilla softmax cross-entropy",
        "objective": (
            r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} "
            r"\log \frac{e^{W_{y_i}^\top x_i + b_{y_i}}}"
            r"{\sum_{j=1}^{C} e^{W_j^\top x_i + b_j}}"
        ),
        "geometry": (
            "Separable but not discriminative: decision boundaries are placed "
            "wherever they reduce training error, with no explicit constraint on "
            "the intra-class radius. Embeddings are not normalised, so the "
            "cosine scoring used at trial time is mismatched with the training "
            "metric."
        ),
        "hyperparams": [],
        "n_classes_dep": True,
        "batch_dep": False,
        "test_normalize": True,
        "predicted_effect": (
            "Weakest of the classification losses. The train/test metric "
            "mismatch (unnormalised inner product vs cosine) should cost the "
            "most on the corpus with the widest duration spread, i.e. slr127."
        ),
    },
    # ------------------------------------------------------------ amsoftmax
    "amsoftmax": {
        "family": "classification",
        "display": "Additive-margin softmax (CosFace)",
        "objective": (
            r"\mathcal{L} = -\frac{1}{N}\sum_i \log "
            r"\frac{e^{s(\cos\theta_{y_i} - m)}}"
            r"{e^{s(\cos\theta_{y_i}-m)} + \sum_{j\neq y_i} e^{s\cos\theta_j}}"
        ),
        "geometry": (
            "Subtracts a constant margin in COSINE space, so the required "
            "angular separation varies with theta: the effective angular margin "
            "m/sin(theta) is small for well-classified samples near theta=0 and "
            "large near the boundary. Enforces cos(theta_y) - cos(theta_j) > m."
        ),
        "hyperparams": ["margin", "scale"],
        "n_classes_dep": True,
        "batch_dep": False,
        "test_normalize": True,
        "predicted_effect": (
            "Close to AAM. The cosine-space margin is gentler than AAM's angular "
            "one at small theta, which should make it more forgiving when the "
            "class count is large relative to the embedding dimension -- i.e. it "
            "should lose less than AAM in the 782-class combined condition."
        ),
    },
    # ----------------------------------------------------------- aamsoftmax
    "aamsoftmax": {
        "family": "classification",
        "display": "Additive-angular-margin softmax (ArcFace)",
        "objective": (
            r"\mathcal{L} = -\frac{1}{N}\sum_i \log "
            r"\frac{e^{s\cos(\theta_{y_i} + m)}}"
            r"{e^{s\cos(\theta_{y_i}+m)} + \sum_{j\neq y_i} e^{s\cos\theta_j}}"
        ),
        "geometry": (
            "Adds the margin to the ANGLE, giving a constant geodesic margin on "
            "the unit hypersphere independent of theta. The maximum number of "
            "classes separable with angular margin m in dimension d is bounded "
            "by the hypersphere packing limit; when C grows past that bound the "
            "margin becomes unsatisfiable and gradients fight each other. This "
            "is the mechanism that makes the si-only (336 class) / ta-only (446) "
            "/ combined (782) comparison informative rather than a data-volume "
            "effect."
        ),
        "hyperparams": ["margin", "scale"],
        "n_classes_dep": True,
        "batch_dep": False,
        "test_normalize": True,
        "predicted_effect": (
            "Best single classification loss in the per-language conditions. In "
            "the combined condition the packing pressure at C=782 may erode the "
            "advantage unless nOut is raised or the margin lowered -- which is "
            "exactly what the Stage C margin sweep measures."
        ),
    },
    # ------------------------------------------------------------ angleproto
    "angleproto": {
        "family": "metric",
        "display": "Angular prototypical",
        "objective": (
            r"S_{j,k} = w\cdot\cos(x_{j,q}, c_k) + b,\quad "
            r"\mathcal{L} = -\frac{1}{N}\sum_j \log "
            r"\frac{e^{S_{j,j}}}{\sum_k e^{S_{j,k}}}"
            r"\quad\text{with}\quad c_k=\frac{1}{M-1}\sum_{m=1}^{M-1}x_{k,m}"
        ),
        "geometry": (
            "Builds class centroids from the batch itself rather than learning "
            "them, so it has no (nOut x nClasses) weight matrix and its capacity "
            "does not grow with speaker count. The learnable scale w and bias b "
            "calibrate the cosine into a logit. Directly optimises the cosine "
            "similarity used at trial time -- train and test metrics match."
        ),
        "hyperparams": ["nPerSpeaker (>=2)"],
        "n_classes_dep": False,
        "batch_dep": True,
        "test_normalize": True,
        "predicted_effect": (
            "The key hypothesis of this study. Because it is class-count-free, "
            "angleproto should degrade LEAST when moving to the 782-class "
            "combined condition, and should be the most robust on the held-out "
            "corpora (open-set, unseen speakers) where a learned centroid matrix "
            "carries no useful information."
        ),
    },
    # ----------------------------------------------------------------- ge2e
    "ge2e": {
        "family": "metric",
        "display": "Generalized end-to-end",
        "objective": (
            r"S_{ji,k} = w\cdot\cos(x_{ji}, c_k) + b,\quad "
            r"\mathcal{L} = -\frac{1}{N}\sum_{j,i} \log "
            r"\frac{e^{S_{ji,j}}}{\sum_k e^{S_{ji,k}}}"
        ),
        "geometry": (
            "As angular prototypical, but every utterance acts as a query "
            "against all centroids (with the query excluded from its own "
            "centroid), so it extracts more constraints per batch at higher "
            "memory cost."
        ),
        "hyperparams": ["nPerSpeaker (>=2)"],
        "n_classes_dep": False,
        "batch_dep": True,
        "test_normalize": True,
        "predicted_effect": (
            "Similar to angleproto with slightly lower variance per step, but "
            "more sensitive to batch composition; expected to need the larger "
            "nPerSpeaker to pay off."
        ),
    },
    # ---------------------------------------------------------------- proto
    "proto": {
        "family": "metric",
        "display": "Prototypical (squared-Euclidean)",
        "objective": (
            r"\mathcal{L} = -\frac{1}{N}\sum_j \log "
            r"\frac{e^{-\|x_{j,q}-c_j\|_2^2}}{\sum_k e^{-\|x_{j,q}-c_k\|_2^2}}"
        ),
        "geometry": (
            "Euclidean rather than angular. Note test_normalize is FALSE for "
            "this loss, so the embedding norm carries information at training "
            "time that cosine scoring then discards -- a train/test metric "
            "mismatch of the same kind as vanilla softmax."
        ),
        "hyperparams": ["nPerSpeaker (>=2)"],
        "n_classes_dep": False,
        "batch_dep": True,
        "test_normalize": False,
        "predicted_effect": (
            "Should trail angleproto for exactly the metric-mismatch reason; "
            "included to isolate 'angular vs Euclidean' from 'metric vs "
            "classification', since it shares the prototypical structure."
        ),
    },
    # ---------------------------------------------------------- softmaxproto
    "softmaxproto": {
        "family": "hybrid",
        "display": "Softmax + angular prototypical (joint)",
        "objective": r"\mathcal{L} = \mathcal{L}_{\text{softmax}} + \mathcal{L}_{\text{angleproto}}",
        "geometry": (
            "Sums a class-anchored term and a batch-relative term. The softmax "
            "branch supplies a global reference frame that pure metric losses "
            "lack, while the prototypical branch keeps the objective aligned "
            "with cosine scoring. Standard strong baseline in the voxceleb "
            "trainer literature."
        ),
        "hyperparams": ["nPerSpeaker (>=2)", "nClasses"],
        "n_classes_dep": True,
        "batch_dep": True,
        "test_normalize": True,
        "predicted_effect": (
            "Expected overall winner on the per-language conditions -- it gets "
            "the global frame from softmax and metric alignment from angleproto. "
            "Its class-count dependence should make it lose ground to pure "
            "angleproto in the combined condition."
        ),
    },
    # -------------------------------------------------------------- triplet
    "triplet": {
        "family": "metric",
        "display": "Triplet with (optional) hard-negative mining",
        "objective": (
            r"\mathcal{L} = \frac{1}{N}\sum_i "
            r"\max\left(0,\ d(a_i,p_i) - d(a_i,n_i) + m\right)"
        ),
        "geometry": (
            "Only constrains sampled triplets, so the gradient signal per batch "
            "is far sparser than the prototypical losses, and the result depends "
            "heavily on the mining strategy (hard_rank / hard_prob). Included as "
            "the classical metric-learning reference point."
        ),
        "hyperparams": ["margin", "hard_rank", "hard_prob", "nPerSpeaker (>=2)"],
        "n_classes_dep": False,
        "batch_dep": True,
        "test_normalize": True,
        "predicted_effect": (
            "Weakest metric loss; slow convergence and mining-sensitive. Present "
            "so the study can state the prototypical family's advantage against "
            "a measured baseline rather than by assertion."
        ),
    },
}

# Losses that read across the batch need several utterances per speaker.
METRIC_LOSSES = {k for k, v in LOSS_SPECS.items() if v["batch_dep"]}
CLASS_COUNT_LOSSES = {k for k, v in LOSS_SPECS.items() if v["n_classes_dep"]}


def get(name: str) -> dict:
    """Return the spec for `name`, raising a helpful error if unknown."""
    if name not in LOSS_SPECS:
        raise KeyError(f"unknown loss {name!r}; known: {sorted(LOSS_SPECS)}")
    return dict(LOSS_SPECS[name])


def required_n_per_speaker(name: str) -> int:
    """Minimum --nPerSpeaker for this loss to be well-defined.

    The batch-structured losses build a centroid from M-1 support utterances and
    query with the M-th, so M must be at least 2. Passing M=1 silently produces a
    degenerate objective rather than an error, which is the kind of bug that
    quietly wastes a GPU-day, so the registry asserts on it.
    """
    return 2 if name in METRIC_LOSSES else 1
