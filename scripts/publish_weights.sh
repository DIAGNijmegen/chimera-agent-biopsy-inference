#!/usr/bin/env bash
#
# Publish the HViT model weights as a GitHub release, so the Docker build can
# fetch them at build time (keeps weights out of git — no Git LFS). Run this once
# per weights version, from a checkout that has the .pt files staged under
# hvit/checkpoints/pretrained/ and hvit/checkpoints/trained/ (both gitignored).
#
# Requires the GitHub CLI authenticated with write access:  gh auth login
set -euo pipefail

REPO="${WEIGHTS_REPO:-DIAGNijmegen/chimera-agent-biopsy-inference}"
TAG="${WEIGHTS_RELEASE:-weights-v1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
files=(
  "${ROOT}"/hvit/checkpoints/pretrained/vit_256_small_dino_fold_{0,1,2,3,4}.pt
  "${ROOT}"/hvit/checkpoints/trained/fold-{0,1,2,3,4}.pt
)

# 1. the local weights must match the committed checksums before we publish them
for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "missing weight file: $f" >&2; exit 1; }
done
( cd "$ROOT" && sha256sum -c resources/SHA256SUMS )

# 2. create the release if needed, then (re)upload the assets + checksums
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$TAG" --repo "$REPO" \
    --title "HViT model weights (${TAG})" \
    --notes "HViT biopsy ISUP weights: per-fold DINO ViT feature extractors (vit_256_small_dino) + trained MIL ensemble heads, 5 CV folds. Fetched at Docker build time; see README."
fi

gh release upload "$TAG" --repo "$REPO" --clobber \
  "${files[@]}" "${ROOT}/resources/SHA256SUMS"

echo "Published to https://github.com/${REPO}/releases/tag/${TAG}"
