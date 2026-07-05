#!/usr/bin/env bash
# Download DCASE Task 2 datasets (2023-2026), backbone checkpoints, and the
# official evaluator repositories. Idempotent: re-running resumes/verifies.
#
# Layout created:
#   ~/data/dcase{2023,2024,2025,2026}t2/{dev,eval}/{zips,raw}
#   ~/models/{beats,panns}
#   ~/tools/dcase{2023,2024,2025}_task2_evaluator
set -u

fetch_record() {  # <zenodo-record-id> <dest-dir>
  local rec="$1" dir="$2"
  mkdir -p "$dir"; cd "$dir" || return 1
  curl -s --retry 5 "https://zenodo.org/api/records/$rec" |
    python3 -c "import json,sys; [print(f['key']) for f in json.load(sys.stdin)['files']]" |
  while read -r f; do
    wget -q -c "https://zenodo.org/records/$rec/files/$f?download=1" -O "$f"
  done
}

extract_all() {  # <zips-dir> <raw-dir>
  mkdir -p "$2"; cd "$2" || return 1
  local z
  for z in "$1"/*.zip; do
    python3 -m zipfile -e "$z" . || echo "failed to extract $z"
  done
}

D=~/data

# --- DCASE 2025 (dev 15097779; additional train 15392814; eval 15519362) ---
fetch_record 15097779 "$D/dcase2025t2/dev/zips"
fetch_record 15392814 "$D/dcase2025t2/eval/zips"
fetch_record 15519362 "$D/dcase2025t2/eval/zips"
extract_all "$D/dcase2025t2/dev/zips"  "$D/dcase2025t2/dev/raw"
extract_all "$D/dcase2025t2/eval/zips" "$D/dcase2025t2/eval/raw"

# --- DCASE 2023 (7882613; 7830345; 7860847) ---
fetch_record 7882613 "$D/dcase2023t2/dev/zips"
fetch_record 7830345 "$D/dcase2023t2/eval/zips"
fetch_record 7860847 "$D/dcase2023t2/eval/zips"
extract_all "$D/dcase2023t2/dev/zips"  "$D/dcase2023t2/dev/raw"
extract_all "$D/dcase2023t2/eval/zips" "$D/dcase2023t2/eval/raw"

# --- DCASE 2024 (10902294; 11259435; 11363076) ---
fetch_record 10902294 "$D/dcase2024t2/dev/zips"
fetch_record 11259435 "$D/dcase2024t2/eval/zips"
fetch_record 11363076 "$D/dcase2024t2/eval/zips"
extract_all "$D/dcase2024t2/dev/zips"  "$D/dcase2024t2/dev/raw"
extract_all "$D/dcase2024t2/eval/zips" "$D/dcase2024t2/eval/raw"

# --- DCASE 2026 development set (19336329; two-channel audio) ---
fetch_record 19336329 "$D/dcase2026t2/dev/zips"
extract_all "$D/dcase2026t2/dev/zips" "$D/dcase2026t2/dev/raw"

# --- Backbone checkpoints (SHA-256 pins in release/checkpoints.json) ---
mkdir -p ~/models/beats ~/models/panns
wget -q -c "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt" \
  -O ~/models/beats/BEATs_iter3_plus_AS2M.pt
wget -q -c "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1" \
  -O ~/models/panns/Cnn14_16k_mAP=0.438.pth
# EAT downloads automatically from Hugging Face on first use.

# --- Official evaluators (post-challenge ground truth for 2023-2025) ---
for y in 2023 2024 2025; do
  [ -d ~/tools/dcase${y}_task2_evaluator ] ||
    git clone -q --depth 1 "https://github.com/nttcslab/dcase${y}_task2_evaluator" \
      ~/tools/dcase${y}_task2_evaluator
done

echo "DONE. Verify checkpoint hashes against release/checkpoints.json:"
sha256sum ~/models/beats/BEATs_iter3_plus_AS2M.pt ~/models/panns/Cnn14_16k_mAP=0.438.pth
