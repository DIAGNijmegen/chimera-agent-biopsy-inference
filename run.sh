#!/usr/bin/env bash
#
# CHIMERA Agent — Task 2: ISUP grade from H&E biopsy WSIs.
#
# Offline pipeline (all weights baked into the image at build time):
#   1. slide2vec — tile each WSI @ 0.5 mpp and extract per-fold panda-vit-s region features
#   2. aggregator — 5-fold MIL ensemble -> per-slide ISUP grade + latents
#
# by @clementgrisi
set -euo pipefail

APP_DIR=/opt/app

display_help() {
   echo "ISUP Grade Estimation from H&E Biopsy WSIs, by @clementgrisi"
   echo
   echo "Syntax: docker run <image> [-f csv_file] [-o output_folder] [-c config_file]"
   echo "options:"
   echo "  -f   CSV manifest listing the slides to process (required)"
   echo "  -o   output folder (required)"
   echo "  -c   slide2vec configuration file (optional; defaults to the baked-in slide2vec-config.yaml)"
   echo "  -h   show this help and exit"
   echo
}

while getopts ":f:o:c:h" opt; do
  case $opt in
    h) display_help; exit 0 ;;
    f) csv="$OPTARG" ;;
    o) output_folder="$OPTARG" ;;
    c) config_file="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; display_help; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; display_help; exit 1 ;;
  esac
done

if [[ -z "${csv:-}" || -z "${output_folder:-}" ]]; then
    echo "Error: arguments -f and -o are required." >&2
    display_help
    exit 1
fi

config="${APP_DIR}/slide2vec-config.yaml"
if [[ -n "${config_file:-}" ]]; then
  config="$config_file"
fi

mkdir -p "$output_folder"
cp "${csv}" "${output_folder}/manifest.csv"

# ---------------------------------------------------------------------------
# Stage 1 — slide2vec: tiling + per-fold panda-vit-s region feature extraction
#   The five folds share fold-0's tiling coordinates; only the feature
#   extractor weights differ per fold (vit_256_small_dino_fold_${fold}).
# ---------------------------------------------------------------------------
cd "${APP_DIR}/slide2vec"
folds=(0 1 2 3 4)
for fold in "${folds[@]}"; do
  if [ "$fold" -eq 0 ]; then
    python3 slide2vec/main.py \
        --config-file "${config}" \
        --skip-datetime \
        csv="${csv}" \
        visualize=true \
        fold="${fold}"
  else
    python3 slide2vec/main.py \
        --config-file "${config}" \
        --skip-datetime \
        csv="${csv}" \
        visualize=false \
        fold="${fold}" \
        tiling.read_coordinates_from="output/fold-0/coordinates"
  fi
done

# hand the slide2vec tiling artefacts back to the caller
cp -r output/fold-0/*.csv "${output_folder}/."
cp -r output/fold-0/coordinates "${output_folder}/."
cp -r output/fold-0/visualization "${output_folder}/."

# ---------------------------------------------------------------------------
# Stage 2 — aggregator: 5-fold MIL ensemble -> ISUP grade + per-slide latents
# ---------------------------------------------------------------------------
cd "${APP_DIR}"
python3 aggregator/inference.py --config-name "panda-inference" test_csv="${csv}"

# collect outputs
cp aggregator/output/inference/results/submission.csv "${output_folder}/inference.csv"
mkdir -p "${output_folder}/latents"
cp -r aggregator/output/inference/results/latents/* "${output_folder}/latents/."

echo "Done. Outputs written to ${output_folder}"
