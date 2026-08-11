#!/usr/bin/env python3
"""
P3 (2026-07-03 roadmap) — parameter-efficient fine-tuning of a pretrained
SSL backbone (WavLM) for Sinhala/Tamil speaker verification.

Four arms, selected with --mode:
    frozen  backbone frozen; train layer-weights + ASP pooling + AAM head only
            (linear-probe baseline)
    lora    LoRA adapters (peft) on attention projections; backbone base
            weights frozen; head trains          <- literature-recommended arm
    full    everything trains (CNN feature extractor stays frozen);
            separate encoder/head learning rates
    llrd    full fine-tuning with layer-wise LR decay
            (top transformer layer gets --lr_encoder, each lower layer
             multiplied by --llrd_decay)

Architecture: WavLM (hidden states, learnable softmax layer weights)
    -> attentive statistics pooling -> 192-d embedding -> AAM-softmax.
Speaker info concentrates in shallow WavLM layers (WavLM paper), hence the
learnable layer combination rather than last-layer features.

Evaluation: after each epoch, cosine EER on subsampled trial lists
(--eval_subsample pairs per list, seed-stable); the best pooled-EER
checkpoint is kept and evaluated on the FULL lists at the end.

Self-contained on purpose (like tools/zeroshot_eval.py): the compute nodes
do not mount /mnt/ricproject*, so this file + tuneThreshold.py are copied
to shared /home staging and run there.

Example (single arm):
    python peft_finetune.py --mode lora \
        --backbone /home/anuraj/sl_spv_bench/models/wavlm-base-plus \
        --train_list p3_train_list.txt --train_root train_audio \
        --trials test_list_si.txt test_list_ta.txt --eval_root audio \
        --out_dir runs/p3_lora_s42 --seed 42 --device cuda
"""

import argparse
import json
import os
import random
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))        # staged layout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo layout
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf  # noqa: E402

TARGET_SR = 16000


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_audio(path, target_sr=TARGET_SR):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype('float32')
    return audio


class CropTrainSet(Dataset):
    """(label, relpath) rows -> random fixed-length crops."""

    def __init__(self, list_path, root, crop_sec):
        self.root = root
        self.crop = int(crop_sec * TARGET_SR)
        self.rows = []
        with open(list_path) as f:
            for ln in f:
                parts = ln.split()
                if len(parts) == 2:
                    self.rows.append((int(parts[0]), parts[1]))
        self.n_classes = max(l for l, _ in self.rows) + 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        label, rel = self.rows[idx]
        audio = load_audio(os.path.join(self.root, rel))
        if len(audio) <= self.crop:
            pad = self.crop + 1 - len(audio)
            audio = np.concatenate(
                [audio, np.random.randn(pad).astype('float32') * 1e-4])
        start = random.randint(0, len(audio) - self.crop)
        return torch.from_numpy(audio[start:start + self.crop]), label


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class ASP(nn.Module):
    """Attentive statistics pooling (Okabe et al. 2018)."""

    def __init__(self, dim, bottleneck=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, bottleneck), nn.Tanh(), nn.Linear(bottleneck, 1))

    def forward(self, x):                      # x: (B, T, D)
        w = torch.softmax(self.att(x), dim=1)  # (B, T, 1)
        mu = (w * x).sum(dim=1)
        var = (w * x * x).sum(dim=1) - mu * mu
        sd = var.clamp(min=1e-6).sqrt()
        return torch.cat([mu, sd], dim=1)      # (B, 2D)


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax (ArcFace), m=0.2 s=30 repo defaults."""

    def __init__(self, n_classes, emb_dim, margin=0.2, scale=30.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)
        self.m, self.s = margin, scale

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.w))
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, cos.size(1)).to(cos.dtype)
        logits = self.s * (onehot * target + (1 - onehot) * cos)
        return F.cross_entropy(logits, labels), \
            (logits.argmax(dim=1) == labels).float().mean()


class SSLSpeakerNet(nn.Module):
    def __init__(self, backbone_path, n_classes, emb_dim=192):
        super().__init__()
        from transformers import WavLMModel
        self.encoder = WavLMModel.from_pretrained(backbone_path)
        self.encoder.config.output_hidden_states = True
        n_layers = self.encoder.config.num_hidden_layers + 1  # + CNN output
        hid = self.encoder.config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.pool = ASP(hid)
        self.fc = nn.Linear(2 * hid, emb_dim)
        self.head = AAMSoftmax(n_classes, emb_dim)

    def freeze_cnn(self):
        self.encoder.feature_extractor._freeze_parameters()

    def embed(self, wav):                       # wav: (B, S)
        out = self.encoder(wav)
        hs = torch.stack(out.hidden_states, dim=0)        # (L, B, T, D)
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (w * hs).sum(dim=0)                            # (B, T, D)
        return self.fc(self.pool(x))                       # (B, emb)

    def forward(self, wav, labels):
        return self.head(self.embed(wav), labels)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def head_params(model):
    return (list(model.pool.parameters()) + list(model.fc.parameters())
            + list(model.head.parameters()) + [model.layer_weights])


def configure_arm(model, args):
    """Freeze/wrap per --mode; return optimizer param groups + description."""
    model.freeze_cnn()
    if args.mode == 'frozen':
        for p in model.encoder.parameters():
            p.requires_grad = False
        groups = [{'params': head_params(model), 'lr': args.lr_head}]
    elif args.mode == 'lora':
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=0.05,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        model.encoder = get_peft_model(model.encoder, cfg)
        lora_params = [p for p in model.encoder.parameters() if p.requires_grad]
        groups = [{'params': lora_params, 'lr': args.lr_lora},
                  {'params': head_params(model), 'lr': args.lr_head}]
    elif args.mode == 'full':
        groups = [{'params': [p for p in model.encoder.parameters()
                              if p.requires_grad], 'lr': args.lr_encoder},
                  {'params': head_params(model), 'lr': args.lr_head}]
    elif args.mode == 'llrd':
        layers = model.encoder.encoder.layers
        n = len(layers)
        groups = []
        for i, layer in enumerate(layers):     # top layer -> lr_encoder
            lr = args.lr_encoder * (args.llrd_decay ** (n - 1 - i))
            groups.append({'params': list(layer.parameters()), 'lr': lr})
        rest = [p for name, p in model.encoder.named_parameters()
                if p.requires_grad and not name.startswith(
                    ('encoder.layers', 'feature_extractor'))
                and 'layers.' not in name]
        if rest:
            groups.append({'params': rest,
                           'lr': args.lr_encoder * (args.llrd_decay ** n)})
        groups.append({'params': head_params(model), 'lr': args.lr_head})
    else:
        raise ValueError(args.mode)
    n_train = sum(p.numel() for g in groups for p in g['params'])
    n_total = sum(p.numel() for p in model.parameters())
    desc = (f"mode={args.mode}: {n_train/1e6:.2f}M trainable "
            f"/ {n_total/1e6:.1f}M total ({100*n_train/n_total:.1f}%)")
    return groups, desc


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def read_trials(path, subsample=0, seed=42):
    trials = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) == 3:
                trials.append((int(parts[0]), parts[1], parts[2]))
    if subsample and subsample < len(trials):
        rng = random.Random(seed)
        tgt = [t for t in trials if t[0] == 1]
        imp = [t for t in trials if t[0] == 0]
        k_t = max(1, int(round(subsample * len(tgt) / len(trials))))
        trials = rng.sample(tgt, k_t) + rng.sample(imp, subsample - k_t)
    return trials


@torch.no_grad()
def evaluate(model, trial_sets, eval_root, device, max_sec=8.0):
    model.eval()
    files = sorted({f for ts in trial_sets.values() for _l, a, b in ts
                    for f in (a, b)})
    cap = int(max_sec * TARGET_SR)
    embs = {}
    for rel in files:
        wav = load_audio(os.path.join(eval_root, rel))[:cap]
        if len(wav) < TARGET_SR // 2:
            wav = np.concatenate(
                [wav, np.zeros(TARGET_SR // 2 - len(wav), dtype='float32')])
        t = torch.from_numpy(wav).unsqueeze(0).to(device)
        e = model.embed(t).squeeze(0)
        embs[rel] = F.normalize(e, dim=0).cpu().numpy()
    out = {}
    for name, trials in trial_sets.items():
        scores = [float(np.dot(embs[a], embs[b])) for _l, a, b in trials]
        labels = [l for l, _a, _b in trials]
        res = tuneThresholdfromScore(scores, labels, [1, 0.1])
        fnrs, fprs, ths = ComputeErrorRates(scores, labels)
        dcf01, _ = ComputeMinDcf(fnrs, fprs, ths, 0.01, 1, 1)
        dcf05, _ = ComputeMinDcf(fnrs, fprs, ths, 0.05, 1, 1)
        out[name] = {'eer_avg': round(res[5], 4), 'eer_max': round(res[1], 4),
                     'mindcf_p01': round(dcf01, 4), 'mindcf_p05': round(dcf05, 4),
                     'n_trials': len(trials)}
    model.train()
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="P3 PEFT fine-tuning (WavLM).")
    p.add_argument('--mode', required=True,
                   choices=['frozen', 'lora', 'full', 'llrd'])
    p.add_argument('--backbone', required=True,
                   help="HF id or local dir (safetensors)")
    p.add_argument('--train_list', required=True)
    p.add_argument('--train_root', required=True)
    p.add_argument('--trials', nargs='+', required=True)
    p.add_argument('--eval_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--crop_sec', type=float, default=2.0)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_encoder', type=float, default=1e-5)
    p.add_argument('--lr_lora', type=float, default=5e-4)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--llrd_decay', type=float, default=0.9)
    p.add_argument('--lr_gamma', type=float, default=0.95)
    p.add_argument('--eval_subsample', type=int, default=2000,
                   help="pairs per list for per-epoch eval; full lists at end")
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit_batches', type=int, default=0,
                   help="debug: cap batches per epoch")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = CropTrainSet(args.train_list, args.train_root, args.crop_sec)
    print(f"[p3] train: {len(ds)} utts, {ds.n_classes} classes", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.num_workers > 0)

    model = SSLSpeakerNet(args.backbone, ds.n_classes).to(args.device)
    groups, desc = configure_arm(model, args)
    print(f"[p3] {desc}", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler(enabled=args.device.startswith('cuda'))

    sub_trials = {os.path.basename(t): read_trials(t, args.eval_subsample,
                                                   args.seed)
                  for t in args.trials}
    full_trials = {os.path.basename(t): read_trials(t) for t in args.trials}

    history, best = [], {'pooled_eer': 1e9}
    for epoch in range(1, args.epochs + 1):
        t0, tot_loss, tot_acc, nb = time.time(), 0.0, 0.0, 0
        for wav, labels in dl:
            wav = wav.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda',
                                    enabled=args.device.startswith('cuda')):
                loss, acc = model(wav, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g['params']], 5.0)
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item(); tot_acc += acc.item(); nb += 1
            if args.limit_batches and nb >= args.limit_batches:
                break
        sched.step()
        ev = evaluate(model, sub_trials, args.eval_root, args.device)
        pooled = float(np.mean([m['eer_avg'] for m in ev.values()]))
        row = {'epoch': epoch, 'loss': round(tot_loss / max(nb, 1), 4),
               'train_acc': round(tot_acc / max(nb, 1), 4),
               'pooled_eer': round(pooled, 4), 'eval': ev,
               'sec': round(time.time() - t0, 1)}
        history.append(row)
        print(f"[p3] epoch {epoch}: loss {row['loss']} acc {row['train_acc']} "
              f"pooled_EER {pooled:.3f}% ({row['sec']}s) "
              f"{ {k: v['eer_avg'] for k, v in ev.items()} }", flush=True)
        if pooled < best['pooled_eer']:
            best = {'pooled_eer': pooled, 'epoch': epoch}
            torch.save({'state_dict': model.state_dict(),
                        'args': vars(args), 'epoch': epoch},
                       os.path.join(args.out_dir, 'best.pt'))
        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
            json.dump({'args': vars(args), 'trainable': desc,
                       'history': history, 'best': best}, f, indent=2)

    # Final: reload best checkpoint, evaluate on FULL trial lists.
    ckpt = torch.load(os.path.join(args.out_dir, 'best.pt'),
                      map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    final = evaluate(model, full_trials, args.eval_root, args.device)
    with open(os.path.join(args.out_dir, 'final_eval.json'), 'w') as f:
        json.dump({'best_epoch': ckpt['epoch'], 'mode': args.mode,
                   'seed': args.seed, 'final': final}, f, indent=2)
    print(f"[p3] FINAL (best epoch {ckpt['epoch']}, full lists): "
          f"{ {k: v['eer_avg'] for k, v in final.items()} }", flush=True)


if __name__ == '__main__':
    main()
