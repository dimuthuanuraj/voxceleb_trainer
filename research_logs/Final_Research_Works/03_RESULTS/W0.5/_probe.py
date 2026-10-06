
import json, sys, os
sys.path.insert(0, '/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/experiments')
import registry as r

res = {}
res["registered"] = [k for k in r.FRONTENDS if k.endswith("_lw_ft")]

# The arbiter must differ from ssl_wavlm_ft on exactly one thing: the model.
a, b = r.FRONTENDS["ssl_wavlm_lw_ft"], r.FRONTENDS["ssl_wavlm_ft"]
diff = sorted(k for k in set(a) | set(b)
              if k not in ("note", "family", "deviations") and a.get(k) != b.get(k))
res["differs_from_ssl_wavlm_ft_on"] = diff
res["single_factor"] = diff == ["model"]

# ...and from ssl_wavlm_lw on exactly the freezing/optimiser package.
c = r.FRONTENDS["ssl_wavlm_lw"]
res["differs_from_ssl_wavlm_lw_on"] = sorted(
    k for k in set(a) | set(c)
    if k not in ("note", "family", "deviations") and a.get(k) != c.get(k))

# Build a real experiment and emit its argv -- the actual test.
try:
    exps = r.stage_f(conditions=("si",))
    res["stage_f_builds"] = True
except Exception as exc:
    res["stage_f_builds"] = False
    res["stage_f_error"] = repr(exc)

try:
    e = r.Experiment(exp_id="F_ssl_wavlm_lw_ft_aamsoftmax_si_s42", stage="F",
                     arch_key="ssl_wavlm_lw_ft", loss_key="aamsoftmax",
                     condition_key="si", seed=42)
    argv = e.argv()
    res["argv_ok"] = True
    res["argv"] = argv if isinstance(argv, list) else str(argv)
    res["min_gpu_mb"] = e.min_gpu_mb()
    joined = " ".join(map(str, argv))
    res["has_no_ssl_freeze"] = "--no_ssl_freeze" in joined
    res["has_llrd"] = "--llrd" in joined
    res["model_is_lw"] = "SSLFrontendSpeakerLW" in joined
except Exception as exc:
    res["argv_ok"] = False
    res["argv_error"] = repr(exc)

print(json.dumps(res, indent=2)[:4000])
open('/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/Final_Research_Works/03_RESULTS/W0.5/probe.json', "w").write(json.dumps(res, indent=2))
