# chimera-agent-biopsy-inference

Containerized **ISUP grade** estimation from H&E **prostate biopsy** whole-slide
images, for the
[CHIMERA Agent](https://chimera-agent.grand-challenge.org/) grand challenge
(**Task 2**).

For each case it tiles the WSI(s), extracts HViT region features with
[slide2vec](https://github.com/clemsgrs/slide2vec) (v1.3.0), runs a 5-fold MIL ensemble to
produce a per-slide ISUP grade plus the per-fold slide-level latents, and
post-processes those into a single per-case WSI latent. A whole folder / manifest
of cases is processed in one invocation.

The image is **portable and offline**: both the per-fold feature-extractor
weights and the five trained ensemble heads are baked in at build time, so it
runs with no network access or credentials at runtime.

## Pipeline

```
manifest.csv (slide_id, wsi_path[, mask_path])
  └─ slide2vec: tile @ 0.5 mpp → panda-vit-s region features (.pt per case, per fold)
     └─ HViT MIL ensemble ×5 folds → per-fold slide latent + ISUP prediction (inference.csv)
        └─ post-process: concat the 5 per-fold [192] latents → single [960] latent (.pt + .json per case)
```

Three stages, all run inside the container by `run.sh`:

1. **slide2vec** — tile each WSI @ 0.5 mpp and extract per-fold `panda-vit-s`
   region features. The five folds share fold-0's tiling coordinates; only the
   feature-extractor weights differ per fold.
2. **aggregator** — the 5-fold HViT MIL ensemble produces a per-fold slide-level
   `[192]` latent and per-fold ISUP prediction, plus the majority-vote ensemble
   grade in `inference.csv`.
3. **post-process** — concatenate the five per-fold `[192]` latents into the
   single `[960]` deliverable latent per case and write the JSON companion.

## Model

- **Feature extractor:** `panda-vit-s` via slide2vec — a hierarchical DINO ViT
  (`vit_256_small_dino`) applied at the region level (tiling at 0.5 mpp, 2048 px
  regions unrolled into 256 px patches, 384-dim features). One feature-extractor
  checkpoint per fold.
- **Aggregator:** Hierarchical ViT MIL head (`region-size 2048`,
  `embed-dim 192`), 5 CV folds, trained as ISUP regression (`num_classes 6`).
  Each fold predicts a grade; the per-case grade is a custom-distance majority
  vote across folds.

The HViT model family is described in our paper, published in
[*Medical Image Analysis*](https://www.sciencedirect.com/science/article/pii/S1361841525002105).

## Prerequisites

- NVIDIA GPU + driver, Docker, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`--gpus all`).
- **Network access at build time only** — the weights are fetched during the
  build and baked in; the image then runs fully offline.
- Slide formats are whatever the base image's ASAP/OpenSlide stack supports
  (`.mrxs`, `.tif`, `.svs`, …).

## Build

```bash
docker build -t chimera-biopsy-inference .
```

> **Maintainers:** to (re)publish the weights release from a checkout that has
> the `.pt` files staged under `aggregator/checkpoints/pretrained/` and
> `aggregator/checkpoints/trained/`, run `bash scripts/publish_weights.sh` (needs repo
> write access).

## Run

```bash
docker run --gpus all \
  -v /path/to/slides:/slides \
  -v /path/to/masks:/masks \
  -v /path/to/output:/output_folder \
  -v /path/to/manifest.csv:/manifest.csv \
  chimera-biopsy-inference \
    -f /manifest.csv \
    -o /output_folder
```

- `-f` **(required)** — CSV manifest of slides to process.
- `-o` **(required)** — output folder (mounted, writable).
- `-c` *(optional)* — alternative slide2vec config; defaults to the baked-in
  top-level `slide2vec-config.yaml`.

Manifest paths must resolve **inside the container**, so mount the slide
directory and write the manifest paths accordingly.

> **Tissue masks.** If a row has no `mask_path`, slide2vec falls back to its
> built-in HSV tissue segmentation, which can under-segment faint / pale tissue
> and silently drop tiles. Prefer supplying a pre-computed mask via `mask_path`;
> otherwise check the saved `‹output_folder›/visualization/` overlays before
> trusting the output.

## I/O contract

**Input** — a CSV manifest in slide2vec format:

| column      | required | meaning                                                  |
|-------------|----------|----------------------------------------------------------|
| `slide_id`  | yes      | case id; keys the prediction and names the latent files. |
| `wsi_path`  | yes      | path to the WSI (resolvable inside the container).       |
| `mask_path` | no       | path to a pre-computed tissue mask (recommended).        |

```csv
slide_id,wsi_path,mask_path
case_001,/slides/case_001.tif,/masks/case_001.tif
```

**Output** — written under the `-o` folder:

```
<output_folder>/
├── manifest.csv                          # copy of the manifest actually used
├── coordinates/                          # slide2vec tiling coordinates
├── visualization/                        # tissue-mask + tiling overlays
├── inference.csv                         # deliverable: per-case ISUP grade
└── latents/
    ├── <slide_id>.pt                     # deliverable: single [960] float32 WSI latent
    ├── <slide_id>.json                   # JSON companion to the .pt
    └── folds/fold-{0..4}/<slide_id>.pt   # per-fold [192] latents (provenance)
```

`inference.csv` — one row per `slide_id`: the per-fold predictions
(`pred_fold-0` … `pred_fold-4`) and the ensemble `pred` (majority-vote ISUP grade,
integer `0..5`).

`latents/<slide_id>.pt` — the five per-fold `[192]` latents flattened and
concatenated in fold order **0,1,2,3,4** into one 1-D `float32` tensor of length
**960**. `latents/<slide_id>.json` carries the same values in the grand-challenge
feature-vector format `[{"title": "", "features": [<960 floats>]}]`.

## Layout

```
.
├── Dockerfile
├── run.sh                       # entrypoint: slide2vec → 5-fold HViT ensemble → post-process
├── slide2vec-config.yaml        # slide2vec tiling + panda-vit-s config (0.5 mpp, region level)
├── requirements.txt             # extra HViT deps (base image provides torch)
├── scripts/publish_weights.sh   # one-time: upload the weights to a GitHub release
├── resources/
│   └── SHA256SUMS               # checksums the build verifies the weights against
├── slide2vec/                   # vendored tiling + panda-vit-s feature extraction, trimmed to that path
└── aggregator/                  # vendored MIL ensemble, trimmed to the inference path
    ├── inference.py             # 5-fold MIL ensemble entrypoint
    ├── postprocess.py           # concat per-fold latents → single [960] .pt + .json
    ├── config/inference/panda-inference.yaml
    ├── source/
    └── tests/                   # pytest unit tests (no GPU / weights needed)
```

The per-fold weights — `aggregator/checkpoints/pretrained/vit_256_small_dino_fold_{0..4}.pt`
(feature extractors) and `aggregator/checkpoints/trained/fold-{0..4}.pt` (ensemble
heads) — are gitignored and fetched from the weights release at build time.

## Provenance

Standalone, offline repackaging of an internal training-pipeline image.
[slide2vec](https://github.com/clemsgrs/slide2vec) (feature extraction) and the
HViT aggregator are vendored here, each trimmed to the inference path.
</content>
</invoke>
