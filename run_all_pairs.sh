#!/usr/bin/env bash
# --------------------------------------------------------------------
# Generate stylised images for all content × style pairs
#   • with AdaIN    → outputs_adain/
#   • without AdaIN → outputs_noadain/
# Saves colour‑harmonised results (…_harmonized.jpg) under standard
# name  <cnt>_stylized_<sty>.png  so evaluation matches.
# --------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StyleID
set -e

REPO=$HOME/StyleID
SCRIPT="$REPO/diffusers_implementation/run_styleid_diffusers.py"
CNT_DIR="$REPO/data/cnt"
STY_DIR="$REPO/data/sty"

OUT_ADAIN="$REPO/outputs_adain"
OUT_NOADAIN="$REPO/outputs_noadain"
LOG_DIR="$REPO/logs"
mkdir -p "$OUT_ADAIN" "$OUT_NOADAIN" "$LOG_DIR"

# common params
GAMMA=0.9
TVAL=1.8
SD_VERSION="2.1-base"
DDIM=20

function run_pair() {
  local cnt=$1  sty=$2  variant=$3  outdir=$4  adain_flag=$5
  local cnt_id=$(basename "$cnt" .png)
  local sty_id=$(basename "$sty" .png)
  local outfile="$outdir/${cnt_id}_stylized_${sty_id}.png"
  local log="$LOG_DIR/${cnt_id}_${sty_id}_${variant}.log"

  [[ -f "$outfile" ]] && { echo "✓ $outfile"; return; }

  echo "→ $cnt_id × $sty_id  [$variant]"
  PYTHONPATH=$PYTHONPATH:. \
  python "$SCRIPT" \
    --cnt_fn "$cnt" \
    --sty_fn "$sty" \
    --gamma "$GAMMA" \
    --T "$TVAL" \
    --ddim_steps "$DDIM" \
    --sd_version "$SD_VERSION" \
    --save_dir "$outdir/tmp" \
    $adain_flag \
    >"$log" 2>&1

  mv "$outdir/tmp/stylized_image_harmonized.jpg" "$outfile"
  rm -rf "$outdir/tmp"
}

for cnt in "$CNT_DIR"/*.png; do
  for sty in "$STY_DIR"/*.png; do
    run_pair "$cnt" "$sty" "adain"    "$OUT_ADAIN"    ""                
    run_pair "$cnt" "$sty" "noadain"  "$OUT_NOADAIN"  "--without_init_adain"
    
    
  done
done

echo "All pairs finished.  Outputs in $OUT_ADAIN and $OUT_NOADAIN"
