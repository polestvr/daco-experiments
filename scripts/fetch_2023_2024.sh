#!/usr/bin/env bash
# Idempotent fetch + verify + extract for DCASE 2023/2024 Task 2.
set -u

fetch_record() {
  local rec="$1" dir="$2"
  mkdir -p "$dir"; cd "$dir" || return 1
  local files
  files=$(curl -s --retry 5 "https://zenodo.org/api/records/$rec" |
    python3 -c "import json,sys; [print(f['key']) for f in json.load(sys.stdin)['files']]") || {
      echo "failed to list files for record $rec"; return 1; }
  local f
  for f in $files; do
    wget -q -c "https://zenodo.org/records/$rec/files/$f?download=1" -O "$f"
  done
}

fetch_record 7882613  ~/data/dcase2023t2/dev/zips
fetch_record 7830345  ~/data/dcase2023t2/eval/zips
fetch_record 7860847  ~/data/dcase2023t2/eval/zips
fetch_record 10902294 ~/data/dcase2024t2/dev/zips
fetch_record 11259435 ~/data/dcase2024t2/eval/zips
fetch_record 11363076 ~/data/dcase2024t2/eval/zips
echo "downloads done"

bad=0
for z in ~/data/dcase2023t2/{dev,eval}/zips/*.zip \
         ~/data/dcase2024t2/{dev,eval}/zips/*.zip; do
  python3 - "$z" <<'PY' || { echo "corrupt zip: $z"; bad=1; }
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).testzip()
PY
done
[ "$bad" = 1 ] && echo "some zips failed verification" || echo "all zips verified"

for y in 2023 2024; do
  for s in dev eval; do
    mkdir -p ~/data/dcase${y}t2/$s/raw
    cd ~/data/dcase${y}t2/$s/raw || continue
    for z in ../zips/*.zip; do
      python3 -m zipfile -e "$z" . || echo "failed to extract $z"
    done
  done
done
echo "extraction done"

[ -d ~/tools/dcase2023_task2_evaluator ] || \
  git clone -q --depth 1 https://github.com/nttcslab/dcase2023_task2_evaluator ~/tools/dcase2023_task2_evaluator
[ -d ~/tools/dcase2024_task2_evaluator ] || \
  git clone -q --depth 1 https://github.com/nttcslab/dcase2024_task2_evaluator ~/tools/dcase2024_task2_evaluator
ls ~/tools/
for y in 2023 2024; do
  echo "== $y dev machines:"; ls ~/data/dcase${y}t2/dev/raw/ 2>/dev/null
  echo "== $y eval machines:"; ls ~/data/dcase${y}t2/eval/raw/ 2>/dev/null
  echo "== $y wav count:"; find ~/data/dcase${y}t2 -name "*.wav" | wc -l
done
echo "done"
