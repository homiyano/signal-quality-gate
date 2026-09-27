#!/usr/bin/env python3
"""
Signal Quality Gate (SQG) for speech-based Alzheimer's disease detection (ADReSSo).
Master's thesis, RQ5 -- Seyedali (Homi) Divbandroudbaraki, University of Rostock.

The gate MEASURES the acoustic quality of every recording, AUDITS whether any quality
descriptor is confounded with the diagnostic label, DECIDES (accept / flag / reject)
with thresholds fitted on training data only, and VALIDATES itself with the
non-speech "shortcut" test of Gauder et al. (2024).  It never modifies the audio.

Sub-commands (run in this order)
--------------------------------
  inspect   print the dataset structure (folders, file counts, formats, CSV headers)
  measure   per-recording quality metrics                      -> metrics.csv
  audit     AD vs. CN test for every metric (confound audit)     -> confound_audit.csv
  gate      accept / flag / reject decisions + exclusion audit   -> gate_decisions.csv
  shortcut  non-speech vs. speech shortcut test (AUC)            -> shortcut_test.csv

Examples
--------
  python signal_quality_gate.py inspect  --data_root ~/data/ADReSSo21
  python signal_quality_gate.py measure  --data_root ~/data/ADReSSo21/diagnosis/train --out sqg_out
  python signal_quality_gate.py audit    --out sqg_out
  python signal_quality_gate.py gate     --out sqg_out [--config gate_config.json] [--fit_ids train_ids.txt]
  python signal_quality_gate.py shortcut --data_root ~/data/ADReSSo21/diagnosis/train --out sqg_out

Labels are taken from the name of the folder that contains each audio file
('ad' / 'cn', case-insensitive).  Segmentation CSVs (speaker, begin, end) are matched
to audio by file stem anywhere below --data_root; they are optional.

Required : numpy scipy pandas soundfile librosa scikit-learn
Optional : pyloudnorm (EBU R128 loudness), speechmos (DNSMOS), silero-vad + torch (VAD)
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
AUDIO_EXT = {".wav", ".mp3", ".flac"}
ANALYSIS_SR = 16000  # what wav2vec2 / Whisper actually see

# ----------------------------------------------------------------------------- optional deps
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None
_SILERO = None
DEVICE = "cpu"          # set by --device; "cuda" moves VAD, resampling and DNSMOS to the GPU
_DNSMOS = None          # lazily created OrtDNSMOS instance (per process)
_DNSMOS_ERR = ""


def setup_runtime(device="cpu", workers=1):
    """Select CPU/GPU for this process. Falls back to CPU with a warning if CUDA is absent."""
    global DEVICE
    DEVICE = "cpu"
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                DEVICE = "cuda"
            else:
                print("WARNING: --device cuda requested but torch.cuda.is_available() is False; using CPU")
        except ImportError:
            print("WARNING: --device cuda requested but PyTorch is not installed; using CPU")
    if workers > 1:
        try:
            import torch
            torch.set_num_threads(1)  # avoid thread oversubscription across worker processes
        except ImportError:
            pass


class OrtDNSMOS:
    """DNSMOS P.835 (Reddy et al., 2021) with explicit ONNX Runtime providers, so it runs on
    CUDA with onnxruntime-gpu. Uses the model files shipped with the `speechmos` package."""
    SR, WIN = 16000, 9.01

    def __init__(self, device="cpu"):
        import importlib.util
        import onnxruntime as ort
        spec = importlib.util.find_spec("speechmos")
        if spec is None:
            raise ImportError("speechmos not installed (needed for the DNSMOS model files)")
        base = Path(spec.origin).parent / "dnsmos_models"
        prov = ["CPUExecutionProvider"]
        if device == "cuda" and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()  # use the CUDA/cuDNN libraries installed with PyTorch
            except Exception:
                pass
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(base / "sig_bak_ovr.onnx"), providers=prov)
        self.providers = self.sess.get_providers()
        self.p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self.p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        self.p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

    def __call__(self, y):
        y = np.asarray(y, np.float32)
        n = int(self.WIN * self.SR)
        while len(y) < n:
            y = np.concatenate([y, y])
        hops = int(np.floor(len(y) / self.SR) - self.WIN) + 1
        sig, bak, ovr = [], [], []
        for i in range(hops):
            # slicing identical to Microsoft's reference dnsmos_local.py (incl. its float rounding)
            seg = y[int(i * self.SR): int((i + self.WIN) * self.SR)]
            if len(seg) < n:
                continue
            raw = self.sess.run(None, {"input_1": seg[np.newaxis, :]})[0][0]
            sig.append(self.p_sig(raw[0])); bak.append(self.p_bak(raw[1])); ovr.append(self.p_ovr(raw[2]))
        return float(np.mean(sig)), float(np.mean(bak)), float(np.mean(ovr))


def get_dnsmos():
    global _DNSMOS, _DNSMOS_ERR
    if _DNSMOS is None and not _DNSMOS_ERR:
        try:
            _DNSMOS = OrtDNSMOS(DEVICE)
        except Exception as e:
            _DNSMOS_ERR = f"{type(e).__name__}: {e}"
    return _DNSMOS


def _lazy_imports():
    global sf, librosa, signal, stats
    import soundfile as sf
    import librosa
    from scipy import signal, stats


# ============================================================================ 0. INSPECT
def cmd_inspect(a):
    root = Path(a.data_root).expanduser()
    if not root.exists():
        sys.exit(f"Not found: {root}")
    print(f"Dataset root: {root}\n")
    dirs = defaultdict(Counter)
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            dirs[str(p.parent.relative_to(root))][p.suffix.lower() or "<none>"] += 1
    print("Folder tree (files per extension):")
    for d in sorted(dirs):
        print(f"  {d:<55} " + ", ".join(f"{k}:{v}" for k, v in sorted(dirs[d].items())))

    csvs = sorted(root.rglob("*.csv"))
    shown = set()
    for c in csvs:  # one example header per folder
        if c.parent in shown:
            continue
        shown.add(c.parent)
        try:
            df = pd.read_csv(c, nrows=3)
            print(f"\nCSV example {c.relative_to(root)}\n  columns: {list(df.columns)}\n"
                  + df.head(3).to_string(index=False).replace("\n", "\n  "))
        except Exception as e:
            print(f"\nCSV {c.name}: unreadable ({e})")

    try:
        import soundfile as sf
    except ImportError:
        print("\n(soundfile not installed: skipping audio header summary)")
        return
    rows = []
    for p in sorted(q for q in root.rglob("*") if q.suffix.lower() in AUDIO_EXT):
        try:
            i = sf.info(str(p))
            rows.append((p.parent.name, i.samplerate, i.channels, i.subtype, i.duration))
        except Exception:
            rows.append((p.parent.name, None, None, "unreadable", None))
    if rows:
        df = pd.DataFrame(rows, columns=["folder", "sr", "channels", "subtype", "dur"])
        print("\nAudio header summary (per folder):")
        print(df.groupby(["folder", "sr", "channels", "subtype"], dropna=False)
                .agg(n=("dur", "size"), mean_dur_s=("dur", "mean"), max_dur_s=("dur", "max"))
                .round(1).to_string())


# ============================================================================ helpers
def find_audio(root):
    files = sorted(p for p in Path(root).expanduser().rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit("No audio files found under --data_root")
    return files


def find_segmentations(root):
    """Every CSV below root keyed by file stem; only stems that match an audio file are used."""
    return {p.stem: p for p in Path(root).expanduser().rglob("*.csv")}


_LABEL_MAP = {"ad": "ad", "probablead": "ad", "probable ad": "ad", "dementia": "ad",
              "cn": "cn", "control": "cn", "hc": "cn"}


def load_label_table(root, extra_csv=None):
    """id -> 'ad'/'cn' from any label CSV below root (ADReSSo: task1.csv for the test set,
    adresso-train-mmse-scores.csv for train) plus an optional --labels_csv."""
    table = {}
    cands = list(Path(root).expanduser().rglob("*.csv"))
    if extra_csv:
        cands.append(Path(extra_csv).expanduser())
    for c in cands:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        cols = {x.lower().strip(): x for x in df.columns}
        idc = next((cols[k] for k in ("id", "adressfname") if k in cols), None)
        dxc = next((cols[k] for k in ("dx",) if k in cols), None)
        if idc is None or dxc is None or "speaker" in cols:
            continue
        for i, d in zip(df[idc].astype(str), df[dxc].astype(str)):
            lab = _LABEL_MAP.get(d.strip().lower())
            if lab and i.strip():
                table.setdefault(i.strip(), lab)
    return table


def label_of(path, table):
    par = path.parent.name.lower()
    if par in ("ad", "cn"):
        return par
    return table.get(path.stem, "unknown")


def split_of(path):
    s = str(path).lower()
    return "test" if "test" in s else "train"


def read_segmentation(path, duration_s):
    """Return list of (speaker, start_s, end_s). Handles ms or s, flexible column names."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    spk = next((cols[c] for c in cols if "speaker" in c or c == "spk"), None)
    beg = next((cols[c] for c in cols if c in ("begin", "start", "onset", "tmin")), None)
    end = next((cols[c] for c in cols if c in ("end", "stop", "offset", "tmax")), None)
    if not (spk and beg and end):
        return []
    b, e = df[beg].astype(float).values, df[end].astype(float).values
    scale = 1000.0 if e.max() > duration_s * 5 else 1.0  # ms -> s
    return [(str(s).strip().upper(), bb / scale, ee / scale) for s, bb, ee in zip(df[spk], b, e)]


def load_audio(path):
    """Native-rate mono + 16 kHz mono, plus header information."""
    info = sf.info(str(path))
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)       # (n, ch)
    ch_corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] == 2 and len(x) > 10 else np.nan
    mono = x.mean(axis=1)
    return mono, sr, resample16(mono, sr), info, ch_corr


def resample16(mono, sr):
    if sr == ANALYSIS_SR:
        return mono.astype(np.float32)
    if DEVICE == "cuda":
        try:
            import torch, torchaudio.functional as AF
            t = torch.from_numpy(np.ascontiguousarray(mono)).to("cuda")
            return AF.resample(t, sr, ANALYSIS_SR).cpu().numpy().astype(np.float32)
        except Exception:
            pass
    return librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR).astype(np.float32)


def vad_mask(y, sr=ANALYSIS_SR, hop_s=0.01):
    """Frame-level speech mask (10 ms hop). Silero if available, else energy-based."""
    global _SILERO
    n_frames = int(np.ceil(len(y) / (sr * hop_s)))
    try:
        import torch
        if _SILERO is None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            model = load_silero_vad()
            dev = "cpu"
            if DEVICE == "cuda":
                try:
                    model = model.to("cuda"); dev = "cuda"
                    get_speech_timestamps(torch.zeros(16000, device="cuda"), model, sampling_rate=sr)
                except Exception:
                    model = load_silero_vad(); dev = "cpu"   # this Silero build is CPU-only
            _SILERO = (model, get_speech_timestamps, dev)
        model, gst, dev = _SILERO
        ts = gst(torch.from_numpy(y).to(dev), model, sampling_rate=sr, return_seconds=True)
        m = np.zeros(n_frames, bool)
        for t in ts:
            m[int(t["start"] / hop_s):int(np.ceil(t["end"] / hop_s))] = True
        return m, "silero"
    except Exception:
        hop = int(sr * hop_s)
        rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.025), hop_length=hop)[0][:n_frames]
        e = 20 * np.log10(rms + EPS)
        m = e > np.percentile(e, 10) + 12.0
        return np.pad(m, (0, n_frames - len(m))), "energy"


def frames_to_samples(m, n, sr=ANALYSIS_SR, hop_s=0.01):
    s = np.repeat(m, int(sr * hop_s))
    return np.pad(s, (0, max(0, n - len(s))), mode="edge")[:n]


def interval_mask(intervals, n_frames, hop_s=0.01):
    m = np.zeros(n_frames, bool)
    for _, b, e in intervals:
        m[int(b / hop_s):int(np.ceil(e / hop_s))] = True
    return m


# ============================================================================ 1. MEASURE
def measure_one(path, seg_path, label):
    mono, sr, y, info, ch_corr = load_audio(path)
    dur = len(mono) / sr
    r = {"id": path.stem, "label": label, "split": split_of(path), "native_sr": sr,
         "channels": info.channels, "subtype": info.subtype, "duration_s": round(dur, 2),
         "stereo_channel_corr": ch_corr}

    # --- level / integrity -------------------------------------------------
    r["peak_dbfs"] = 20 * np.log10(np.max(np.abs(mono)) + EPS)
    # clipping = samples inside runs of >= 3 consecutive samples at >= 99.9% of the file's peak
    a_ = np.abs(mono); hit = a_ >= 0.999 * (a_.max() + EPS)
    if hit.any():
        d = np.diff(np.r_[0, hit.astype(np.int8), 0]); st_, en_ = np.where(d == 1)[0], np.where(d == -1)[0]
        ln = en_ - st_; r["clip_ratio"] = float(ln[ln >= 3].sum() / len(mono))
    else:
        r["clip_ratio"] = 0.0
    r["dc_offset"] = float(np.mean(mono))
    r["loudness_lufs"] = np.nan
    if pyln is not None and dur > 1:
        try:
            r["loudness_lufs"] = pyln.Meter(sr).integrated_loudness(mono.astype(np.float64))
        except Exception:
            pass

    # --- spectral integrity (native rate) ----------------------------------
    f, p = signal.welch(mono, fs=sr, nperseg=4096)
    pdb = 10 * np.log10(p + EPS)
    above = np.where(pdb > pdb.max() - 60)[0]
    r["bandwidth_hz"] = float(f[above[-1]]) if len(above) else 0.0
    if sr >= 32000:  # 14-16 kHz band: codec / bit-rate-mode cue (Gauder et al., 2024)
        hf = (f >= 14000) & (f <= 16000)
        r["hf_14_16k_rel_db"] = float(10 * np.log10(p[hf].mean() / (p.mean() + EPS) + EPS))
    else:
        r["hf_14_16k_rel_db"] = np.nan

    # --- speech / noise (16 kHz) -------------------------------------------
    vad, vad_kind = vad_mask(y)
    s = frames_to_samples(vad, len(y))
    ps = np.mean(y[s] ** 2) if s.any() else EPS
    pn = np.mean(y[~s] ** 2) if (~s).any() else EPS
    r["vad"] = vad_kind
    r["speech_ratio"] = float(vad.mean())
    r["snr_vad_db"] = float(10 * np.log10((ps + EPS) / (pn + EPS)))
    r["noise_floor_dbfs"] = float(10 * np.log10(pn + EPS))

    # --- DNSMOS (non-intrusive MOS; Reddy et al., 2021) ----------------------
    for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"):
        r[k] = np.nan
    dm = get_dnsmos()
    if dm is not None:
        yy = y / (np.max(np.abs(y)) + EPS)
        r["dnsmos_sig"], r["dnsmos_bak"], r["dnsmos_ovrl"] = dm(yy)

    # --- speaker structure (segmentation CSV or RTTM-converted CSV) ---------
    for k in ("par_speech_s", "inv_speech_s", "overlap_ratio", "par_snr_db"):
        r[k] = np.nan
    r["par_source"] = "none"
    if seg_path is not None:
        segs = read_segmentation(seg_path, dur)
        if segs:
            n = len(vad)
            par = interval_mask([t for t in segs if t[0].startswith("PAR")], n)
            inv = interval_mask([t for t in segs if t[0].startswith("INV")], n)
            r["par_source"] = "segmentation"
            if not par.any():  # segmentation has no PAR turns: VAD speech outside INV turns
                par = vad[:n] & ~inv
                r["par_source"] = "vad_minus_inv"
            r["par_speech_s"] = round(par.sum() * 0.01, 2)
            r["inv_speech_s"] = round(inv.sum() * 0.01, 2)
            r["overlap_ratio"] = float((par & inv).sum() / max(par.sum(), 1))
            ps_ = frames_to_samples(par & ~inv, len(y))
            if ps_.any():
                r["par_snr_db"] = float(10 * np.log10(np.mean(y[ps_] ** 2) / (pn + EPS) + EPS))
    return r


def _worker_init(device, workers):
    _lazy_imports()
    setup_runtime(device, workers)


def _measure_task(args):
    p, seg, lab = args
    try:
        r = measure_one(p, seg, lab)
    except Exception as e:
        r = {"id": p.stem, "label": lab, "split": split_of(p), "error": f"{type(e).__name__}: {e}"}
    r["_dnsmos_err"] = _DNSMOS_ERR
    return r


def run_parallel(fn, tasks, a, label):
    """Run fn over tasks in order, with --workers processes (spawned, CUDA-safe)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    out, n = [None] * len(tasks), len(tasks)
    if a.workers <= 1:
        for i, t in enumerate(tasks):
            out[i] = fn(t)
            print(f"\r[{label} {i + 1}/{n}]", end="", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                                    initializer=_worker_init, initargs=(a.device, a.workers)) as ex:
            futs = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
            for k, f in enumerate(cf.as_completed(futs), 1):
                out[futs[f]] = f.result()
                print(f"\r[{label} {k}/{n}]", end="", flush=True)
    print()
    return out


def cmd_measure(a):
    _lazy_imports()
    setup_runtime(a.device, 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files, segs = find_audio(a.data_root), find_segmentations(a.data_root)
    labels = load_label_table(a.data_root, a.labels_csv)
    print(f"Device: {DEVICE}   workers: {a.workers}   files: {len(files)}")
    if a.workers <= 1:
        dm = get_dnsmos()
        print("DNSMOS providers:", dm.providers if dm else f"OFF ({_DNSMOS_ERR})")
    rows = run_parallel(_measure_task, [(p, segs.get(p.stem), label_of(p, labels)) for p in files], a, "measure")
    errs = {r.pop("_dnsmos_err", "") for r in rows} - {""}
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    print("Recordings per split x label:\n", pd.crosstab(df["split"], df["label"]))
    n_dm = int(df["dnsmos_ovrl"].notna().sum()) if "dnsmos_ovrl" in df else 0
    if errs:
        print(f"WARNING: DNSMOS not computed ({'; '.join(sorted(errs))})")
    if "error" in df and df["error"].notna().any():
        print(f"WARNING: {int(df['error'].notna().sum())} files failed; see the 'error' column")
    print(f"\nWrote {out / 'metrics.csv'}  ({len(df)} recordings, DNSMOS on {n_dm}, "
          f"LUFS={'on' if pyln else 'off'})")


# ============================================================================ 2. AUDIT
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, p[i] * (m - rank))); adj[i] = run
    return adj


def cmd_audit(a):
    from scipy import stats
    out = Path(a.out)
    df = pd.read_csv(out / "metrics.csv")
    if a.split and "split" in df:
        df = df[df["split"] == a.split]
    df = df[df["label"].isin(["ad", "cn"])]
    print(f"Auditing {len(df)} labelled recordings (split='{a.split or 'all'}')")
    num = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 1]
    res = []
    for c in num:
        ad, cn = df.loc[df.label == "ad", c].dropna(), df.loc[df.label == "cn", c].dropna()
        if len(ad) < 5 or len(cn) < 5:
            continue
        u, p = stats.mannwhitneyu(ad, cn, alternative="two-sided")
        auc = u / (len(ad) * len(cn))  # P(metric_AD > metric_CN)
        res.append({"metric": c, "n_ad": len(ad), "n_cn": len(cn), "median_ad": ad.median(),
                    "median_cn": cn.median(), "auc_ad_gt_cn": auc,
                    "effect_r": 2 * auc - 1, "p": p})
    r = pd.DataFrame(res)
    r["p_holm"] = holm(r["p"].values)
    r["confounded"] = r["p_holm"] < 0.05
    r.sort_values("p").to_csv(out / "confound_audit.csv", index=False)
    # categorical: codec / sample rate / channels vs. label
    for c in ("native_sr", "channels", "subtype"):
        if c in df:
            tab = pd.crosstab(df[c], df["label"])
            if tab.shape[0] > 1:
                chi2, p, *_ = stats.chi2_contingency(tab)
                print(f"\n{c} x label (chi2 p={p:.3g}):\n{tab}")
    print("\nConfound audit (Holm-corrected):")
    print(r.sort_values("p")[["metric", "median_ad", "median_cn", "auc_ad_gt_cn", "p_holm",
                              "confounded"]].round(3).to_string(index=False))


# ============================================================================ 3. GATE
DEFAULT_CONFIG = {
    "_comment": "Absolute rules are fixed a priori; percentile rules are fitted on --fit_ids only.",
    "reject": [
        {"metric": "native_sr", "op": "<", "value": 16000},
        {"metric": "snr_vad_db", "op": "<", "value": 0.0},
        {"metric": "par_speech_s", "op": "<", "value": 10.0}
    ],
    "flag": [
        {"metric": "clip_ratio", "op": ">", "value": 0.001},
        {"metric": "dnsmos_ovrl", "op": "<", "percentile": 10},
        {"metric": "snr_vad_db", "op": "<", "percentile": 10},
        {"metric": "overlap_ratio", "op": ">", "percentile": 90},
        {"metric": "par_snr_db", "op": "<", "percentile": 10},
        {"metric": "par_source", "op": "==", "value": "vad_minus_inv"}
    ]
}


def _threshold(rule, fit):
    if "value" in rule:
        return rule["value"]
    col = fit[rule["metric"]].dropna()
    return float(np.percentile(col, rule["percentile"])) if len(col) else np.nan


def cmd_gate(a):
    from scipy import stats
    out = Path(a.out)
    df = pd.read_csv(out / "metrics.csv")
    cfg = json.loads(Path(a.config).read_text()) if a.config else DEFAULT_CONFIG
    fit = df
    if a.fit_split and "split" in df:
        fit = df[df["split"] == a.fit_split]
        print(f"Thresholds fitted on the {len(fit)} '{a.fit_split}' recordings only")
    elif a.fit_ids:
        ids = set(Path(a.fit_ids).read_text().split())
        fit = df[df["id"].isin(ids)]
        print(f"Thresholds fitted on {len(fit)} recordings from {a.fit_ids}")
    if not a.fit_split and not a.fit_ids:
        print("WARNING: no --fit_ids given; percentile thresholds use ALL recordings. "
              "Inside cross-validation pass the training-fold ids to avoid leakage.")

    df["decision"], df["reasons"] = "accept", ""
    used = []
    for level in ("flag", "reject"):  # reject overrides flag
        for rule in cfg.get(level, []):
            m = rule["metric"]
            if m not in df or df[m].isna().all():
                continue
            if rule["op"] == "==":
                thr = rule["value"]; hit = df[m].astype(str) == str(thr)
                df.loc[hit, "decision"] = level
                df.loc[hit, "reasons"] += f"{m}=={thr};"
                used.append({"level": level, "metric": m, "op": "==", "threshold": thr, "n_hit": int(hit.sum())})
                continue
            thr = _threshold(rule, fit)
            hit = (df[m] < thr) if rule["op"] == "<" else (df[m] > thr)
            hit &= df[m].notna()
            df.loc[hit, "decision"] = level
            df.loc[hit, "reasons"] += f"{m}{rule['op']}{thr:.3g};"
            used.append({"level": level, "metric": m, "op": rule["op"], "threshold": thr,
                         "n_hit": int(hit.sum())})
    df.to_csv(out / "gate_decisions.csv", index=False)
    pd.DataFrame(used).to_csv(out / "gate_thresholds_used.csv", index=False)

    # exclusion audit: does the gate remove one class more than the other?
    tab = pd.crosstab(df["label"], df["decision"])
    print("\nDecision x label:\n", tab)
    if {"ad", "cn"} <= set(df.label):
        for level in ("reject", "flag"):
            if level not in tab:
                continue
            t2 = np.array([[tab.loc[c].get(level, 0), tab.loc[c].sum() - tab.loc[c].get(level, 0)]
                           for c in ("ad", "cn")])
            _, p = stats.fisher_exact(t2)
            print(f"{level:>6} rate AD={t2[0,0]/t2[0].sum():.1%}  CN={t2[1,0]/t2[1].sum():.1%}  "
                  f"Fisher p={p:.3g}  {'<-- class-biased!' if p < 0.05 else '(balanced)'}")


# ============================================================================ 4. SHORTCUT TEST
def region_features(y, mask, sr=ANALYSIS_SR, n_mfcc=20):
    s = frames_to_samples(mask, len(y))
    seg = y[s]
    if len(seg) < sr * 1.0:  # need >= 1 s of material
        return None
    M = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=400, hop_length=160)
    return np.r_[M.mean(1), M.std(1)]


def _shortcut_task(args):
    p, lab = args
    _, _, y, _, _ = load_audio(p)
    vad, _ = vad_mask(y)
    return lab, {"non_speech": region_features(y, ~vad), "speech": region_features(y, vad)}


def cmd_shortcut(a):
    """Gauder et al. (2024) protocol, simplified: MFCC statistics from non-speech vs.
    speech regions -> logistic regression, repeated stratified k-fold AUC.
    A non-speech AUC reliably above 0.5 means the label is recoverable from the recording
    conditions alone. Run on all recordings and on gate-accepted recordings."""
    _lazy_imports()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = None
    if a.accepted_only:
        g = pd.read_csv(out / "gate_decisions.csv")
        keep = set(g.loc[g.decision != "reject", "id"])
    table = load_label_table(a.data_root, a.labels_csv)
    setup_runtime(a.device, 1)
    feats = {"non_speech": [], "speech": []}
    labels = {"non_speech": [], "speech": []}
    tasks = []
    for p in find_audio(a.data_root):
        lab = label_of(p, table)
        if (a.split and split_of(p) != a.split) or lab not in ("ad", "cn") or (keep is not None and p.stem not in keep):
            continue
        tasks.append((p, lab))
    for lab, fv in run_parallel(_shortcut_task, tasks, a, "features"):
        for name, v in fv.items():
            if v is not None:
                feats[name].append(v); labels[name].append(int(lab == "ad"))
    rows = []
    for name in feats:
        X, yl = np.array(feats[name]), np.array(labels[name])
        if len(np.unique(yl)) < 2:
            continue
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=42)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        nj = max(1, a.workers)
        auc = cross_val_score(clf, X, yl, cv=cv, scoring="roc_auc", n_jobs=nj)
        row = {"region": name, "subset": "accepted" if keep else "all", "n": len(yl),
               "auc_mean": auc.mean(), "auc_sd": auc.std(),
               "fold_auc_p2.5": np.percentile(auc, 2.5), "fold_auc_p97.5": np.percentile(auc, 97.5)}
        if a.permutations > 0:
            # null distribution: same repeated CV on label-shuffled data
            rng = np.random.default_rng(42); null = []
            cvp = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
            from joblib import Parallel, delayed
            perms = [rng.permutation(yl) for _ in range(a.permutations)]
            null = Parallel(n_jobs=nj)(delayed(lambda yy: cross_val_score(clf, X, yy, cv=cvp,
                                                                          scoring="roc_auc").mean())(yy)
                                       for yy in perms)
            null = np.array(null)
            row["perm_null_mean"] = null.mean()
            row["perm_p"] = (1 + np.sum(null >= auc.mean())) / (1 + len(null))
        rows.append(row)
    r = pd.DataFrame(rows)
    fn = out / f"shortcut_test_{'accepted' if keep else 'all'}.csv"
    r.to_csv(fn, index=False)
    print("\n" + r.round(3).to_string(index=False))
    print("\nInterpretation: perm_p < 0.05 on the non-speech row = recording conditions predict "
          "the label above chance (acquisition shortcut).")


# ============================================================================ DOCTOR
def cmd_doctor(a):
    """Check the server environment before a long run."""
    import platform
    print("Python", platform.python_version())
    try:
        import torch
        print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
              "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    except ImportError:
        print("torch: NOT installed")
    try:
        import torchaudio
        print("torchaudio", torchaudio.__version__)
    except ImportError:
        print("torchaudio: not installed (GPU resampling off, librosa used)")
    try:
        import onnxruntime as ort
        print("onnxruntime", ort.__version__, "| providers:", ort.get_available_providers())
    except ImportError:
        print("onnxruntime: NOT installed (DNSMOS off)")
    _lazy_imports(); setup_runtime(a.device, 1)
    dm = get_dnsmos()
    print("DNSMOS:", f"ready on {dm.providers}" if dm else f"OFF ({_DNSMOS_ERR})")
    y = (0.1 * np.random.default_rng(0).standard_normal(ANALYSIS_SR * 3)).astype(np.float32)
    m, kind = vad_mask(y)
    print("VAD:", kind, "| device:", _SILERO[2] if _SILERO else "cpu")
    print("Selected device:", DEVICE)


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("--data_root", required=True)
    s = sub.add_parser("measure"); s.add_argument("--data_root", required=True); s.add_argument("--out", default="sqg_out")
    s.add_argument("--labels_csv", help="optional extra id,dx CSV")
    s = sub.add_parser("audit"); s.add_argument("--out", default="sqg_out")
    s.add_argument("--split", default="train", help="train | test | '' (all)")
    s = sub.add_parser("gate"); s.add_argument("--out", default="sqg_out")
    s.add_argument("--config"); s.add_argument("--fit_ids")
    s.add_argument("--fit_split", default="train", help="fit percentile thresholds on this split only ('' = all)")
    s = sub.add_parser("shortcut"); s.add_argument("--data_root", required=True)
    s.add_argument("--out", default="sqg_out"); s.add_argument("--accepted_only", action="store_true")
    s.add_argument("--repeats", type=int, default=10)
    s.add_argument("--permutations", type=int, default=500, help="label permutations for the p-value (0 = off)")
    s.add_argument("--labels_csv"); s.add_argument("--split", default="train", help="train | test | '' (all)")
    s = sub.add_parser("write-config"); s.add_argument("--path", default="gate_config.json")
    s = sub.add_parser("doctor")
    for name, sp in sub.choices.items():
        if name in ("measure", "shortcut", "doctor"):
            sp.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
            sp.add_argument("--workers", type=int, default=1, help="parallel processes (files)")
    a = ap.parse_args()
    if a.cmd == "write-config":
        Path(a.path).write_text(json.dumps(DEFAULT_CONFIG, indent=2)); print(f"Wrote {a.path}")
        return
    {"inspect": cmd_inspect, "measure": cmd_measure, "audit": cmd_audit,
     "gate": cmd_gate, "shortcut": cmd_shortcut, "doctor": cmd_doctor}[a.cmd](a)


if __name__ == "__main__":
    main()
