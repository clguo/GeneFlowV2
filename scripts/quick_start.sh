#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-C1}"
data_root="${DATA_ROOT:-data}"
output="${OUTPUT_DIR:-outputs/$dataset}"
case "$dataset" in C1|C2|P1) ;; *) echo 'Dataset must be C1, C2, or P1' >&2; exit 1 ;; esac
if [[ -f "$data_root/prepared/$dataset/dataset.json" ]]; then
  python -m geneflowv2.datasets check --dataset "$dataset" --data-root "$data_root"
else
  python -m geneflowv2.datasets download --dataset "$dataset" --data-root "$data_root"
fi
python -m geneflowv2.run train --dataset "$dataset" --data-root "$data_root" --output-dir "$output" --auto_resume
python -m geneflowv2.run validate --dataset "$dataset" --data-root "$data_root" --output-dir "$output"
