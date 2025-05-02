#!/usr/bin/env bash
# Batch style-transfer: every content × every style, auto-mask version
# Usage:  bash run_all_automask.sh

set -e  # stop on first error

# Activate your conda env (edit if your env name is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StyleID

# Root dir of repo (directory that contains diffusers_implementation/)
REPO_DIR="$HOME/StyleID"
CNT_DIR="$REPO_DIR/data/cnt"
STY_DIR="$REPO_DIR/data/sty"
SCRIPT="$REPO_DIR/diffusers_implementation/run_styleid_diffusers.py"

OUT_DIR="$REPO_DIR/outputs_automask"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Parameters (tweak if desired)
GAMMA=0.75
TVAL=1.5
DDIM_STEPS=50              # use same defaults as before
SD_VERSION="2.1-base"      # or 1.5 etc.

# Loop
for cnt_path in "$CNT_DIR"/*.png; do
  cnt_id=$(basename "$cnt_path" .png)          # e.g. 04
  for sty_path in "$STY_DIR"/*.png; do
    sty_id=$(basename "$sty_path" .png)        # e.g. 17

    out_file="$OUT_DIR/${cnt_id}_stylized_${sty_id}.png"
    log_file="$LOG_DIR/automask_${cnt_id}_${sty_id}.out"

    # skip if already computed
    if [[ -f "$out_file" ]]; then
      echo "✓  exists  $out_file"
      continue
    fi

    echo "→  $cnt_id  ×  $sty_id  →  $out_file"
    python "$SCRIPT" \
      --cnt_fn "$cnt_path" \
      --sty_fn "$sty_path" \
      --gamma "$GAMMA" \
      --T "$TVAL" \
      --ddim_steps "$DDIM_STEPS" \
      --sd_version "$SD_VERSION" \
      --save_dir "$(dirname "$out_file")" \
      >  "$log_file" 2>&1

    # move the produced stylized_image.jpg to desired filename
    mv "$(dirname "$out_file")/stylized_image.jpg" "$out_file"
  done
done

echo "All pairs done.  Results in  $OUT_DIR/"
