#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Held-out evaluation of a trained experiment, with per-trial scores retained.

    python experiments/tools/evaluate.py --exp A_ecapa1024_aamsoftmax_si_s42
    python experiments/tools/evaluate.py --all --stage A
    python experiments/tools/evaluate.py --exp <id> --probes      # + held-out corpora

Why this is a separate pass rather than the trainer's own ``--eval``
--------------------------------------------------------------------
The trainer reports a scalar EER.  The analysis needs the **per-trial score
vector together with the speaker identity behind each side of each trial**,
because the inference statistic is a speaker-clustered paired bootstrap: it
resamples *speakers*, not trials, and it compares two systems on the *same*
trials.  Neither is reconstructable from a scalar.  So this pass extracts
embeddings once, scores every trial, and writes the raw vectors to disk.

What it produces, per experiment, under ``experiments/results/<exp_id>/``::

    test_eval.json          EER / minDCF / DET points per evaluation set
    scores/<set>.npz        scores, labels, and the speaker id of each side

Evaluation sets
---------------
* the experiment's own held-out test trials (speaker-disjoint from training)
* every held-out *corpus* probe (``--probes``): never trained on by any run, so
  these carry the cross-corpus generalisation claim
* the NISP cross-lingual list, the only genuine same-speaker si/ta-vs-en trials
  available, used to measure the language penalty within a speaker

AS-Norm is evaluated as an explicit on/off factor here rather than during
training, so its contribution is attributable separately from the backbone's.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")


# --------------------------------------------------------------------------
def read_trials(path):
    """Return (labels, side_a, side_b) from a `<label> <path> <path>` list."""
    labels, a, b = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            labels.append(int(parts[0]))
            a.append(parts[1])
            b.append(parts[2])
    # int32, not int8: tuneThreshold.ComputeErrorRates accumulates a running
    # sum in the array's own dtype, which overflows an int8 after 127 targets
    # and silently corrupts every EER it computes.
    return numpy.array(labels, dtype=numpy.int32), a, b


def speaker_of(rel_path):
    """Speaker id from a VoxCeleb-style path.

    Trees here are ``<lang>/<speaker>/<session>/<utt>.wav``; absolute paths from
    the combined condition carry the same tail. Taking the component three from
    the end is stable under both.
    """
    parts = rel_path.replace("\\", "/").rstrip("/").split("/")
    return parts[-3] if len(parts) >= 3 else parts[0]


def extract_embeddings(model, files, root, device, eval_frames, batch=32,
                       num_eval=10, verbose=True):
    """Mean-pooled, L2-normalised embedding per file.

    Each file is split into ``num_eval`` evenly spaced crops of ``eval_frames``
    and the crop embeddings are averaged before normalisation -- the standard
    full-utterance protocol, and the same one the trainer's own evaluation uses,
    so numbers here stay comparable with the training-time curves.
    """
    import torch
    from DatasetLoader import loadWAV

    out = {}
    t0 = time.time()
    for i in range(0, len(files), batch):
        chunk = files[i : i + batch]
        feats = []
        for f in chunk:
            path = f if os.path.isabs(f) else os.path.join(root, f)
            audio = loadWAV(path, eval_frames, evalmode=True, num_eval=num_eval)
            feats.append(torch.FloatTensor(audio))
        with torch.no_grad():
            for f, feat in zip(chunk, feats):
                emb = model(feat.to(device)).detach().cpu()
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out[f] = emb.mean(dim=0, keepdim=True)
        if verbose and (i // batch) % 20 == 0:
            done = min(i + batch, len(files))
            print(f"    embeddings {done}/{len(files)}  "
                  f"({done / max(time.time() - t0, 1e-9):.1f} files/s)", flush=True)
    return out


def cosine_scores(embs, a, b):
    import torch

    A = torch.cat([embs[x] for x in a], dim=0)
    B = torch.cat([embs[x] for x in b], dim=0)
    return torch.nn.functional.cosine_similarity(A, B, dim=1).numpy()


def as_norm_scores(embs, a, b, cohort_embs, top_k=300):
    """Adaptive symmetric score normalisation.

        s_norm(e,t) = 0.5 * [ (s - mu_top(e)) / sd_top(e)
                            + (s - mu_top(t)) / sd_top(t) ]

    where mu/sd are taken over the top-K cohort scores for that side.  AS-Norm
    removes per-utterance score offsets, which is exactly the nuisance that
    makes a single global threshold mis-calibrated across languages.
    """
    import torch

    if not cohort_embs:
        return None
    C = torch.cat(list(cohort_embs.values()), dim=0)
    C = torch.nn.functional.normalize(C, p=2, dim=1)
    k = min(top_k, C.shape[0])

    stats = {}
    uniq = sorted(set(a) | set(b))
    for i in range(0, len(uniq), 256):
        block = uniq[i : i + 256]
        E = torch.cat([embs[x] for x in block], dim=0)
        sims = torch.mm(E, C.t())
        top = torch.topk(sims, k=k, dim=1).values
        mu, sd = top.mean(dim=1), top.std(dim=1).clamp_min(1e-6)
        for j, f in enumerate(block):
            stats[f] = (float(mu[j]), float(sd[j]))

    raw = cosine_scores(embs, a, b)
    out = numpy.empty_like(raw)
    for i, (x, y) in enumerate(zip(a, b)):
        mx, sx = stats[x]
        my, sy = stats[y]
        out[i] = 0.5 * ((raw[i] - mx) / sx + (raw[i] - my) / sy)
    return out


def compute_metrics(scores, labels, p_target=0.05, c_miss=1, c_fa=1):
    from tuneThreshold import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore

    # Plain Python ints and floats: these helpers accumulate in whatever type
    # they are handed, so numpy scalars reintroduce the overflow risk.
    scores = [float(s) for s in scores]
    labels = [int(l) for l in labels]
    res = tuneThresholdfromScore(scores, labels, [1, 0.1])
    fnrs, fprs, thr = ComputeErrorRates(scores, labels)
    mindcf, dcf_thr = ComputeMinDcf(fnrs, fprs, thr, p_target, c_miss, c_fa)
    return {
        "eer": float(res[1]),
        "eer_threshold": float(res[4]),
        "mindcf": float(mindcf),
        "mindcf_threshold": float(dcf_thr),
        "n_trials": int(len(labels)),
        "n_target": int((numpy.asarray(labels) == 1).sum()),
    }


# --------------------------------------------------------------------------
def _coerce(text):
    """Recover an argv string's original type the way argparse would."""
    if text in ("True", "False"):
        return text == "True"
    for cast in (int, float):
        try:
            return cast(text)
        except (TypeError, ValueError):
            pass
    return text


def argv_parameters(res_dir):
    """Model arguments recovered from the literal command that trained the run.

    ``manifest["resolved_parameters"]`` records only the parameters the registry
    knows how to vary, so anything passed straight through to the model is
    absent from it -- every ``--ssl_*`` argument, in particular. Rebuilding from
    the manifest alone therefore silently falls back to the *model's own
    defaults*, and for the SSL front ends those defaults are wrong in two ways:

    * ``ssl_encoder_name`` defaults to the hub id ``microsoft/wavlm-base`` while
      every run actually trained against a local weights directory. Harmless to
      the numbers -- the checkpoint carries all 229 encoder tensors and
      overwrites it -- but it made the load itself fail, because the cached hub
      revision ships only ``pytorch_model.bin`` and transformers >= 4.56 refuses
      ``torch.load`` under torch < 2.6 (CVE-2025-32434).
    * ``ssl_layer`` defaults to ``-1``. The layer index is not a weight, so a
      mismatch loads *cleanly* -- every tensor matches, no warning is printed --
      and the model then reads a different hidden layer than it trained on.
      F_ssl_wavlm_low trained on layer 0 would have been scored on layer 12, the
      worst layer available, and reported a confident wrong EER.

    ``command.json`` is the argv that actually ran, so it is the ground truth.
    """
    path = os.path.join(res_dir, "command.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        argv = json.load(fh)
    out, i = {}, 0
    while i < len(argv):
        tok = argv[i]
        if isinstance(tok, str) and tok.startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if isinstance(nxt, str) and not nxt.startswith("--"):
                out[tok[2:]] = _coerce(nxt)
                i += 2
                continue
            out[tok[2:]] = True
        i += 1
    return out


def trainer_defaults():
    """Every ``--flag`` default declared by ``trainSpeakerNet.py``.

    The trainer passes its *whole* parsed namespace into the model, so an
    argument the experiment never mentions still reaches the model carrying the
    trainer's default -- which is not the same as the model's own default.
    ``--encoder_type`` is the case that bit: trainer default ``SAP``, model
    default ``ASP``. Rebuilding from manifest + argv alone missed it, so every
    SSL model was rebuilt with ASP pooling (1536-dim, mean+std) against a
    checkpoint trained with SAP (768-dim). The 229 encoder tensors matched and
    loaded; ``bn.*`` and ``fc.weight`` did not, leaving the final projection at
    *random initialisation* -- and the run still reported a plausible EER.

    Parsed with ``ast`` rather than imported: importing trainSpeakerNet.py would
    execute it. Parsing is also why this fixes the whole class rather than one
    flag -- any trainer default that differs from a model default is now
    supplied.
    """
    import ast

    path = os.path.join(REPO_ROOT, "trainSpeakerNet.py")
    out = {}
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        name = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                name = str(arg.value)[2:]
                break
        if not name:
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    out[name] = ast.literal_eval(kw.value)
                except Exception:
                    pass
    return out


def load_model(manifest, checkpoint, device, res_dir=None):
    """Rebuild the exact model this experiment trained and load its weights."""
    import torch
    from SpeakerNet import SpeakerNet, WrappedModel

    params = dict(manifest["resolved_parameters"])
    # argv first, manifest second: both describe the same run, but the manifest
    # is the authority on what the registry chose, while argv is the only record
    # of what was passed through to the model.
    if res_dir:
        for k, v in argv_parameters(res_dir).items():
            params.setdefault(k, v)
    # Lowest priority: whatever the trainer's argparse would have supplied for
    # flags nobody passed explicitly.
    for k, v in trainer_defaults().items():
        params.setdefault(k, v)
    kwargs = {k: v for k, v in params.items()
              if k not in ("model", "trainfunc", "optimizer", "scheduler")}
    kwargs.setdefault("nPerSpeaker", params.get("n_per_speaker", 1))
    kwargs.setdefault("nClasses", params.get("n_classes", 1000))
    kwargs.setdefault("nOut", params.get("nOut", 256))

    # SpeakerNet directly, NOT WrappedModel: ModelTrainer.saveParameters writes
    # `self.__model__.module.state_dict()`, so checkpoints carry bare
    # "__S__.*" / "__L__.*" keys. Loading into a WrappedModel would look for
    # "module.__S__.*" and match nothing -- silently producing an untrained
    # network and a plausible-looking chance-level EER.
    net = SpeakerNet(
        model=params["model"],
        optimizer=params.get("optimizer", "adam"),
        trainfunc=params["trainfunc"],
        **kwargs,
    ).to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if len(state.keys()) == 1 and "model" in state:
        state = {"__S__." + k: v for k, v in state["model"].items()}

    own = net.state_dict()
    loaded, skipped = 0, []
    for name, param in state.items():
        key = name if name in own else name.replace("module.", "")
        if key in own and own[key].size() == param.size():
            own[key].copy_(param)
            loaded += 1
        else:
            skipped.append(name)
    net.eval()

    if loaded == 0:
        raise SystemExit(
            f"checkpoint {checkpoint} matched no parameters in the rebuilt model "
            f"-- refusing to report metrics from untrained weights"
        )
    # The loss head holds the class-centroid matrix, which does not transfer to
    # unseen speakers and is not used at scoring time -- skipping it is
    # expected, so only a *backbone* mismatch is worth reporting.
    backbone_skipped = [s for s in skipped if "__L__" not in s]
    if backbone_skipped:
        # Previously a warning. It should never have been: a backbone tensor
        # that fails to match is left at random initialisation, and the run then
        # reports a confident, meaningless EER. That is exactly what happened to
        # nine SSL evaluations -- bn.* and fc.weight silently random because the
        # model was rebuilt with ASP pooling against a SAP checkpoint. A wrong
        # number is worse than no number, so this now refuses to score.
        raise SystemExit(
            f"{len(backbone_skipped)} backbone tensor(s) from {checkpoint} did "
            f"not match the rebuilt model: {backbone_skipped[:6]}\n"
            f"The rebuilt architecture differs from the trained one; scoring it "
            f"would report metrics from partly-random weights. Fix the rebuild "
            f"(see trainer_defaults/argv_parameters) rather than ignoring this."
        )
    return net.__S__, {"tensors_loaded": loaded,
                       "skipped": len(skipped),
                       "backbone_skipped": backbone_skipped[:10]}


def evaluate_set(model, device, name, trials_path, root, eval_frames, cohort=None,
                 cohort_root=None, out_dir=None, as_norm=True):
    labels, a, b = read_trials(trials_path)
    files = sorted(set(a) | set(b))
    print(f"  [{name}] {len(labels)} trials over {len(files)} files")
    embs = extract_embeddings(model, files, root, device, eval_frames)

    scores = cosine_scores(embs, a, b)
    result = {"set": name, "trials": trials_path, "cosine": compute_metrics(scores, labels)}

    norm_scores = None
    if as_norm and cohort:
        # The cohort lives in the TRAINING corpus and is resolved against its
        # own root. Resolving it against `root` would build cohort paths like
        # <tamil_corpus>/wav/si/<sinhala_speaker>/... whenever the evaluation
        # set comes from a different corpus than the model was trained on --
        # which is every transfer and probe set.
        c_root = cohort_root or root
        cohort_embs = {f: embs[f] for f in cohort if f in embs}
        missing = [f for f in cohort if f not in embs]
        if missing:
            cohort_embs.update(
                extract_embeddings(model, missing, c_root, device, eval_frames,
                                   verbose=False)
            )
        norm_scores = as_norm_scores(embs, a, b, cohort_embs)
        if norm_scores is not None:
            result["as_norm"] = compute_metrics(norm_scores, labels)
            result["as_norm"]["cohort_size"] = len(cohort_embs)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        numpy.savez_compressed(
            os.path.join(out_dir, f"{name}.npz"),
            scores=scores,
            scores_asnorm=norm_scores if norm_scores is not None else numpy.array([]),
            labels=labels,
            # Speaker ids are what the bootstrap resamples over.
            spk_a=numpy.array([speaker_of(x) for x in a]),
            spk_b=numpy.array([speaker_of(x) for x in b]),
        )
    return result


def evaluate_experiment(exp_id, probes=False, as_norm=True, device_str=None,
                        checkpoint=None, transfer_en=False):
    import torch

    res_dir = os.path.join(RESULTS_DIR, exp_id)
    man_path = os.path.join(res_dir, "manifest.json")
    if not os.path.isfile(man_path):
        raise SystemExit(f"no manifest for {exp_id} -- has it been run?")
    with open(man_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    ckpt = checkpoint or os.path.join(
        manifest["resolved_parameters"]["save_path"], "model", "model_best.model"
    )
    if not os.path.isfile(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt}")

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, load_info = load_model(manifest, ckpt, device, res_dir=res_dir)
    print(f"[{exp_id}] loaded {load_info['tensors_loaded']} tensors from "
          f"{os.path.relpath(ckpt, REPO_ROOT)} on {device}")

    from experiments import registry as R

    cond = R.CONDITIONS[manifest["experiment"]["condition"]]
    eval_frames = manifest["resolved_parameters"].get("eval_frames", 300)
    scores_dir = os.path.join(res_dir, "scores")

    cohort = None
    if as_norm and cond.get("cohort_list") and os.path.isfile(cond["cohort_list"]):
        with open(cond["cohort_list"], encoding="utf-8") as fh:
            cohort = [l.split()[-1] for l in fh if l.strip()]

    sets = []
    # 1) the experiment's own held-out test trials.
    #    A condition that pools two corpora cannot have one trial list: its file
    #    paths are absolute (test_path "/") because they come from different
    #    roots, and pooling the trials would also hide the very contrast the
    #    condition exists to measure. Those conditions declare
    #    per_lang_test_lists instead and are scored once per subset.
    #    Keyed on the data, not on the condition name: this branch used to test
    #    `condition == "combined"`, which meant si_pooled -- added later, same
    #    shape -- fell through to cond["test_list"] and died with KeyError,
    #    leaving all seven si_pooled runs unevaluable.
    if cond.get("per_lang_test_lists"):
        for subset, path in (p.split(":", 1) for p in
                             cond["per_lang_test_lists"].split(",")):
            sets.append((f"test_{subset}", path, cond.get("test_path", "/")))
    else:
        sets.append(("test", cond["test_list"], cond["test_path"]))

    # 2) cross-language transfer: the OTHER primary corpus, never trained on for
    #    a single-language model. This is the transfer measurement.
    for other_key, other in R.CONDITIONS.items():
        if other_key in ("combined", manifest["experiment"]["condition"]):
            continue
        # Conditions flagged auto_transfer=False (the English pair) are opt-in:
        # they are controls rather than transfer targets, and including them by
        # default would silently double every experiment's evaluation cost.
        if not other.get("auto_transfer", True) and not transfer_en:
            continue
        # Pooled conditions have no single test_list (see above); they are
        # valid training conditions but not transfer targets.
        if not other.get("test_list"):
            continue
        sets.append((f"transfer_{other_key}", other["test_list"], other["test_path"]))

    # 2b) channel-controlled Tamil. slr127 pools three collection sites and
    #     ~48% of its impostor pairs are cross-batch, rejected on channel rather
    #     than on speaker identity (mean score -0.043 vs +0.053; EER 0.475% vs
    #     1.298%). Scoring the same checkpoint against a same-batch-only list
    #     separates the two. No retraining: only the trial list differs.
    sb = os.path.join(R.SPLITS, "slr127_tamil_samebatch", "test_trials.txt")
    if os.path.isfile(sb) and "ta" in cond.get("languages", []):
        sets.append(("test_samebatch", sb, R.CONDITIONS["ta"]["test_path"]))

    # 3) held-out corpora
    if probes:
        for name, probe in R.HELDOUT_PROBES.items():
            if os.path.isfile(probe["trials"]):
                sets.append((f"probe_{name}", probe["trials"], probe["path"]))
        nisp = R.HELDOUT_PROBES["nisp_tamil"].get("cross_lingual_trials")
        if nisp and os.path.isfile(nisp):
            sets.append(("crosslingual_nisp", nisp, R.HELDOUT_PROBES["nisp_tamil"]["path"]))

    results = []
    for name, path, root in sets:
        if not os.path.isfile(path):
            print(f"  [{name}] SKIP -- no trial list at {path}")
            continue
        try:
            results.append(
                evaluate_set(model, device, name, path, root, eval_frames,
                             cohort=cohort, cohort_root=cond.get("cohort_path"),
                             out_dir=scores_dir, as_norm=as_norm)
            )
        except Exception as exc:
            print(f"  [{name}] FAILED: {exc!r}")
            results.append({"set": name, "error": repr(exc)})

    out = {
        "exp_id": exp_id,
        "checkpoint": ckpt,
        "checkpoint_load": load_info,
        "device": str(device),
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sets": results,
    }
    # Never destroy a previous evaluation. Results are not in git, so an
    # overwrite is the only copy gone -- and re-running with a corrected
    # evaluator is exactly when the earlier numbers are most worth keeping, for
    # comparison and for auditing what changed. Prior results are moved aside
    # into eval_history/, stamped with the run that produced them.
    _out = os.path.join(res_dir, "test_eval.json")
    if os.path.isfile(_out):
        _hist = os.path.join(res_dir, "eval_history")
        os.makedirs(_hist, exist_ok=True)
        try:
            with open(_out, encoding="utf-8") as _fh:
                _stamp = json.load(_fh).get("evaluated_utc") or ""
        except Exception:
            _stamp = ""
        _stamp = (_stamp or time.strftime("%Y-%m-%dT%H:%M:%S")).replace(":", "").replace("-", "")
        _dest = os.path.join(_hist, f"test_eval.{_stamp}.json")
        if not os.path.exists(_dest):
            os.rename(_out, _dest)
            print(f"[{exp_id}] previous evaluation kept at "
                  f"{os.path.relpath(_dest, REPO_ROOT)}")

    with open(_out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"[{exp_id}] wrote {os.path.relpath(os.path.join(res_dir, 'test_eval.json'), REPO_ROOT)}")
    for r in results:
        if "cosine" in r:
            line = f"    {r['set']:26s} EER {r['cosine']['eer']:6.3f}%  minDCF {r['cosine']['mindcf']:.4f}"
            if "as_norm" in r:
                line += f"   | AS-Norm EER {r['as_norm']['eer']:6.3f}%"
            print(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", default=None, help="experiment id")
    ap.add_argument("--all", action="store_true", help="every completed experiment")
    ap.add_argument("--stage", default=None, help="restrict --all to one stage")
    ap.add_argument("--probes", action="store_true", help="also run held-out corpora")
    ap.add_argument("--transfer-en", action="store_true",
                    help="also score against the English test sets (adds ~57k "
                         "trials per experiment; off by default because the "
                         "English conditions are controls, not transfer targets)")
    ap.add_argument("--no-as-norm", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", default=None, help="override the checkpoint path")
    ap.add_argument("--redo", action="store_true", help="re-evaluate even if done")
    args = ap.parse_args()

    if args.exp:
        targets = [args.exp]
    elif args.all:
        targets = []
        for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "final.json"))):
            eid = os.path.basename(os.path.dirname(p))
            if eid.endswith("__smoke") or eid.endswith("__dev"):
                continue
            if args.stage and not eid.startswith(f"{args.stage}_"):
                continue
            targets.append(eid)
    else:
        ap.error("pass --exp ID or --all")

    if not targets:
        print("nothing to evaluate")
        return 0

    failed = []
    for eid in targets:
        done = os.path.isfile(os.path.join(RESULTS_DIR, eid, "test_eval.json"))
        if done and not args.redo:
            print(f"[{eid}] already evaluated (--redo to force)")
            continue
        try:
            evaluate_experiment(eid, probes=args.probes,
                                as_norm=not args.no_as_norm,
                                device_str=args.device, checkpoint=args.checkpoint,
                                transfer_en=args.transfer_en)
        except SystemExit as exc:
            print(f"[{eid}] {exc}")
            failed.append(eid)
        except Exception as exc:
            print(f"[{eid}] FAILED: {exc!r}")
            failed.append(eid)
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
