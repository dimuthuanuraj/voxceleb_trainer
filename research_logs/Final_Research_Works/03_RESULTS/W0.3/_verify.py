
import sys, json, torch
sys.path.insert(0, '/mnt/ricproject3/2026/SL_SPV')
sys.path.insert(0, '/mnt/ricproject3/2026/SL_SPV/proposals')
from A5_dk_campp.DK_CAMPP import DK_CAMPP

res = {}
torch.manual_seed(0)
m = DK_CAMPP(nOut=256).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev)
res["device"] = dev

# |x| up to 8.0 -- above the 5.66 observed over 30,000 real augmented samples.
x = (torch.randn(4, 32000, device=dev) * 2.5).clamp(-8.0, 8.0)
res["input_absmax"] = float(x.abs().max())

with torch.no_grad():
    f32 = m.features(x)
res["fp32_finite"] = bool(torch.isfinite(f32).all())

if dev == "cuda":
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        f16 = m.features(x)
    res["amp_dtype"] = str(f16.dtype)
    res["amp_finite"] = bool(torch.isfinite(f16).all())
    res["amp_nonfinite_count"] = int((~torch.isfinite(f16)).sum())
    res["max_abs_diff_vs_fp32"] = float((f16.float() - f32).abs().max())
    # raw power spectrogram headroom, the quantity that actually overflows
    with torch.no_grad():
        p = m.torchfb(x.float())
    res["power_spec_peak"] = float(p.max())
    res["fp16_ceiling"] = 65504.0
    res["headroom_x"] = 65504.0 / max(float(p.max()), 1e-9)
else:
    res["amp_finite"] = None
    res["note"] = "no CUDA on this host: autocast path not exercised. Re-run W0.3 on a GPU node to complete verification."

print(json.dumps(res, indent=2))
open('/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/Final_Research_Works/03_RESULTS/W0.3/verify.json', "w").write(json.dumps(res, indent=2))
