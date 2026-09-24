"""
Two-speaker split with overlap removal for ADReSSo recordings (e.g. adrso025.wav).

Pipeline
  0. Stereo check      -> are the channels spatially different? (Pearson r, RMS ratio)
  1. Preprocess        -> mono downmix (mean of channels), resample 44.1 kHz -> 16 kHz
  2. Diarization       -> pyannote segmentation-3.0 (ONNX) + WeSpeaker ResNet34-LM embeddings
                          + agglomerative clustering (cosine threshold 0.7), via sherpa-onnx
  3. Role verification -> Whisper base.en (ONNX, int8) transcribes every segment
  4. Overlap detection -> intersection of segments that belong to different speakers
  5. Output            -> one time-aligned file per speaker; the other speaker and all
                          overlap regions are muted (150 ms padding, 10 ms fades)

Setup
  pip install sherpa-onnx soundfile librosa numpy
  (models are downloaded automatically from the sherpa-onnx GitHub releases)

Usage
  python split_speakers_pipeline.py adrso025.wav --out outputs/
"""
import argparse, csv, os, tarfile, urllib.request
import numpy as np, soundfile as sf, librosa, sherpa_onnx

REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
MODELS = {
    "seg":  (f"{REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
             "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    "emb":  (f"{REL}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",  # "recongition" is the real tag name
             "wespeaker_en_voxceleb_resnet34_LM.onnx"),
    "asr":  (f"{REL}/asr-models/sherpa-onnx-whisper-base.en.tar.bz2",
             "sherpa-onnx-whisper-base.en"),
}
SR16 = 16000


def fetch(key, mdir):
    url, rel = MODELS[key]
    path = os.path.join(mdir, rel)
    if not os.path.exists(path):
        os.makedirs(mdir, exist_ok=True)
        dst = os.path.join(mdir, os.path.basename(url))
        urllib.request.urlretrieve(url, dst)
        if dst.endswith(".tar.bz2"):
            with tarfile.open(dst) as t:
                t.extractall(mdir)
    return path


def stereo_check(x):
    if x.ndim == 1:
        return "mono file"
    L, R = x[:, 0], x[:, 1]
    r = np.corrcoef(L, R)[0, 1]
    lr_db = 20 * np.log10(np.sqrt((L**2).mean()) / np.sqrt((R**2).mean()))
    return f"channel correlation r={r:.3f}, L/R level={lr_db:+.1f} dB"


def diarize(a16, mdir, threshold=0.7):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=fetch("seg", mdir))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=fetch("emb", mdir)),
        clustering=sherpa_onnx.FastClusteringConfig(threshold=threshold),  # or num_clusters=2
        min_duration_on=0.2,   # drop speech islands shorter than 0.2 s
        min_duration_off=0.3,  # fill pauses shorter than 0.3 s
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return [(s.start, s.end, s.speaker) for s in sd.process(a16).sort_by_start_time()]


def transcribe(a16, segs, mdir):
    d = fetch("asr", mdir)
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=f"{d}/base.en-encoder.int8.onnx", decoder=f"{d}/base.en-decoder.int8.onnx",
        tokens=f"{d}/base.en-tokens.txt", language="en")
    out = []
    for p, q, _ in segs:
        s = rec.create_stream()
        s.accept_waveform(SR16, a16[int(p * SR16):int(q * SR16)])
        rec.decode_stream(s)
        out.append(s.result.text.strip())
    return out


def overlaps(segs):
    ov = []
    for i, (p, q, c) in enumerate(segs):
        for p2, q2, c2 in segs[i + 1:]:
            if c != c2 and p2 < q and q2 > p:
                ov.append((max(p, p2), min(q, q2)))
    return ov


def speaker_mask(segs, spk, n, sr, pad=0.15):
    m = np.zeros(n, bool)
    for p, q, c in segs:                      # own speech + padding
        if c == spk:
            m[int(max(0, p - pad) * sr):int(min(n / sr, q + pad) * sr)] = True
    for p, q, c in segs:                      # remove other speaker -> also removes overlaps
        if c != spk:
            m[int(p * sr):int(q * sr)] = False
    return m


def fade(m, sr, ms=10):
    k = int(sr * ms / 1000)
    return np.convolve(m.astype(np.float32), np.ones(k) / k, "same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.wav))[0]

    x, sr = sf.read(args.wav)
    print("[0]", stereo_check(x))

    mono = x.mean(1) if x.ndim == 2 else x
    a16 = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=SR16)

    segs = diarize(a16, args.models, args.threshold)
    spk_time = {}
    for p, q, c in segs:
        spk_time[c] = spk_time.get(c, 0) + (q - p)
    print("[2] speakers found:", {f"spk{k}": round(v, 1) for k, v in spk_time.items()})
    if len(spk_time) != 2:
        print("    WARNING: not 2 clusters - tune --threshold or use num_clusters=2")

    # Role heuristic for ADReSSo: participant talks most. VERIFY with the transcripts.
    ranked = sorted(spk_time, key=spk_time.get, reverse=True)
    roles = {ranked[0]: "participant", ranked[1]: "interviewer"}

    texts = [""] * len(segs) if args.no_asr else transcribe(a16, segs, args.models)
    ov = overlaps(segs)
    print(f"[4] {len(ov)} overlap regions, {sum(q - p for p, q in ov):.2f} s total")

    with open(os.path.join(args.out, f"{base}_segments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "cluster", "role", "overlap", "transcript"])
        for (p, q, c), t in zip(segs, texts):
            is_ov = any(a < q and b > p for a, b in ov)
            w.writerow([f"{p:.2f}", f"{q:.2f}", f"spk{c}", roles.get(c, "?"), is_ov, t])
            print(f"    {p:7.2f}-{q:7.2f} {roles.get(c, '?'):11s} {'OV' if is_ov else '  '} {t}")

    for spk, role in roles.items():
        g = fade(speaker_mask(segs, spk, len(x), sr), sr)
        y = x * (g[:, None] if x.ndim == 2 else g)
        sf.write(os.path.join(args.out, f"{base}_{role}.wav"), y, sr, subtype="PCM_16")
    print("[5] written to", args.out)


if __name__ == "__main__":
    main()
