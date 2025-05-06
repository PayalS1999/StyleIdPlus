#!/usr/bin/env bash
# ------------------------------------------------------------------
# Parallel StyleID+-generation for all 20×40 pairs
#   • auto‑mask + AdaIN  → outputs_adain/
#   • auto‑mask + no AdaIN → outputs_noadain/
# Skips pairs whose PNG already exists (so you can resume safely).
# Concurrency controlled via CONCURRENCY env var (default 2).
# Requires util/color_harmonize.py and PYTHONPATH fix as before.
# ------------------------------------------------------------------

# 0) Activate env for every subshell --------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StyleID

# 1) Config ---------------------------------------------------------
REPO=$HOME/StyleID
SCRIPT="$REPO/diffusers_implementation/run_styleid_diffusers.py"
CNT_DIR="$REPO/data/cnt"
STY_DIR="$REPO/data/sty"

OUT_ADAIN="$REPO/outputs_adain"
OUT_NOADAIN="$REPO/outputs_noadain"
LOG_DIR="$REPO/logs"
mkdir -p "$OUT_ADAIN" "$OUT_NOADAIN" "$LOG_DIR"

export PYTHONPATH="$PYTHONPATH:$REPO"

# GPU selection -----------------------------------------------------
: ${CUDA_VISIBLE_DEVICES:=0}   # default GPU 0 if user didn’t set it
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

# Parameters for diffusion
GAMMA=0.9
TVAL=1.8
SD_VERSION="2.1-base"
DDIM=20

# Concurrency (parallel jobs)
: ${CONCURRENCY:=2}

# 2) Functions ------------------------------------------------------
run_pair () {
  local cnt_path=$1 sty_path=$2 outdir=$3 adain_flag=$4 variant=$5
  local cnt_id=$(basename "$cnt_path" .png)
  local sty_id=$(basename "$sty_path" .png)
  local outfile="$outdir/${cnt_id}_stylized_${sty_id}.png"
  local log="$LOG_DIR/${cnt_id}_${sty_id}_${variant}.log"

  # skip if finished
  [[ -f "$outfile" ]] && { echo "✓ $outfile"; return 0; }

  # clean tmp dir in case of previous crash
  rm -rf "$outdir/tmp"

  echo "→ $cnt_id × $sty_id  [$variant]"
  PYTHONPATH=$PYTHONPATH \
    python "$SCRIPT" \
      --cnt_fn "$cnt_path" \
      --sty_fn "$sty_path" \
      --gamma "$GAMMA" \
      --T "$TVAL" \
      --ddim_steps "$DDIM" \
      --sd_version "$SD_VERSION" \
      --save_dir "$outdir/tmp" \
      $adain_flag \
      > "$log" 2>&1

  # if success, move harmonised png to final filename
  if [[ -f "$outdir/tmp/stylized_image_harmonized.jpg" ]]; then
    mv "$outdir/tmp/stylized_image_harmonized.jpg" "$outfile"
    echo "★ Done  $outfile"
  else
    echo "✗ Failed $cnt_id $sty_id ($variant) – see $log"
  fi
  rm -rf "$outdir/tmp"
}

# 3) Main loop with job control -------------------------------------
jobcount=0
for cnt in "$CNT_DIR"/*.png; do
  for sty in "$STY_DIR"/*.png; do
    # AdaIN version
    run_pair "$cnt" "$sty" "$OUT_ADAIN" "" "adain" &
    ((jobcount++))

    # no‑AdaIN version
    run_pair "$cnt" "$sty" "$OUT_NOADAIN" "--without_init_adain" "noadain" &
    ((jobcount++))

    # keep number of background jobs ≤ CONCURRENCY
    while [[ $(jobs -r | wc -l) -ge $CONCURRENCY ]]; do
      sleep 1
    done
  done
done
wait
echo "All pairs finished. Outputs in $OUT_ADAIN and $OUT_NOADAIN"
