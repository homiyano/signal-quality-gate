#!/usr/bin/env bash
# Signal Quality Gate -- GPU run for a Linux/CUDA training server.
#
# Usage (from the folder that holds signal_quality_gate.py):
#   bash run_sqg_cuda.sh                              # dataset in ./ADReSSo
#   DATA=/path/to/ADReSSo bash run_sqg_cuda.sh        # dataset elsewhere
#   WORKERS=8 GPU=1 bash run_sqg_cuda.sh              # 8 processes, second GPU
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-ADReSSo}"                      # folder with diagnosis-train/ and diagnosis-test/
OUT="${OUT:-sqg_out}"
WORKERS="${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# 1) isolated environment (created once)
if [ ! -d .venv_cuda ]; then
  python3 -m venv .venv_cuda
fi
source .venv_cuda/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy pandas soundfile librosa scikit-learn pyloudnorm joblib
python -m pip install --quiet torch torchaudio            # Linux wheels ship with CUDA
python -m pip install --quiet silero-vad speechmos onnxruntime-gpu
python -m pip uninstall --quiet -y onnxruntime 2>/dev/null || true   # CPU build would shadow the GPU build

# 2) environment check -- read this output once: CUDA, ONNX providers, DNSMOS, VAD
python signal_quality_gate.py doctor --device cuda | tee "$OUT.doctor.txt"

# 3) the five stages
python signal_quality_gate.py inspect  --data_root "$DATA" | tee "$OUT.inspect.txt"
python signal_quality_gate.py measure  --data_root "$DATA" --out "$OUT" --device cuda --workers "$WORKERS"
python signal_quality_gate.py audit    --out "$OUT" --split train     | tee "$OUT/audit_console.txt"
python signal_quality_gate.py gate     --out "$OUT" --fit_split train | tee "$OUT/gate_console.txt"
python signal_quality_gate.py shortcut --data_root "$DATA" --out "$OUT" --split train \
       --device cuda --workers "$WORKERS"                 | tee "$OUT/shortcut_all_console.txt"
python signal_quality_gate.py shortcut --data_root "$DATA" --out "$OUT" --split train --accepted_only \
       --device cuda --workers "$WORKERS"                 | tee "$OUT/shortcut_accepted_console.txt"

echo
echo "Done. Results are in: $(pwd)/$OUT"
