# CHIMERA Agent — Task 2: ISUP grade from H&E biopsy WSIs.
#
# Self-contained, offline image. The base image provides the slide2vec runtime
# stack; this layer vendors the HViT pipeline, installs the feature-extractor
# dependencies, and bakes in all model weights, so the container runs with no
# network access or credentials at runtime.
FROM waticlems/slide2vec:v1.3.0

# expose ports for optional interactive debugging (ssh / jupyter)
EXPOSE 22 8888

# Everything lives under /opt/app, never /home/user: the enroot/pyxis runtime
# overmounts the home dir, discarding anything baked there during build.
USER root
RUN mkdir -p /opt/app && chown user:user /opt/app
USER user
WORKDIR /opt/app

# vendored slide2vec (tiling + panda-vit-s region feature extraction)
COPY --chown=user:user slide2vec ./slide2vec
# top-level slide2vec feature-extraction config (the default run.sh passes to slide2vec)
COPY --chown=user:user slide2vec-config.yaml ./slide2vec-config.yaml
# vendored aggregator pipeline (5-fold MIL ensemble), trimmed to the inference path;
# the inference config lives at aggregator/config/inference/panda-inference.yaml
COPY --chown=user:user aggregator ./aggregator

# Python deps, installed as root so they land in global site-packages: as USER
# user, pip falls back to a --user install under /home/user/.local, which the
# runtime overmount discards. Build-time network only; baked for offline run.
#   - aggregator requirements (the base image already provides torch/torchvision/numpy)
#   - openslide-bin: the slide reader the biopsy slides need (config backend)
COPY --chown=user:user requirements.txt ./requirements.txt
USER root
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir openslide-bin
USER user

# Aggregator model weights — per-fold DINO ViT feature extractors (pretrained/) and
# trained MIL ensemble heads (trained/). Fetched from a GitHub release at build
# time and verified against resources/SHA256SUMS, so they stay out of git (no
# Git LFS) and the image is self-contained. Override the tag with
# --build-arg WEIGHTS_RELEASE=<tag>.
ARG WEIGHTS_REPO=DIAGNijmegen/chimera-agent-biopsy-inference
ARG WEIGHTS_RELEASE=weights-v1
COPY --chown=user:user resources/SHA256SUMS ./resources/SHA256SUMS
RUN mkdir -p aggregator/checkpoints/pretrained aggregator/checkpoints/trained && \
    base="https://github.com/${WEIGHTS_REPO}/releases/download/${WEIGHTS_RELEASE}" && \
    fetch() { python3 -c "import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2"; } && \
    for f in 0 1 2 3 4; do \
      echo "Downloading vit_256_small_dino_fold_${f}.pt" && \
      fetch "${base}/vit_256_small_dino_fold_${f}.pt" "aggregator/checkpoints/pretrained/vit_256_small_dino_fold_${f}.pt" && \
      echo "Downloading fold-${f}.pt" && \
      fetch "${base}/fold-${f}.pt" "aggregator/checkpoints/trained/fold-${f}.pt"; \
    done && \
    sha256sum -c resources/SHA256SUMS && \
    echo "all weights verified"

# vendored slide2vec/aggregator on the import path
ENV PYTHONPATH="/opt/app/slide2vec:/opt/app/aggregator:${PYTHONPATH}"

# entrypoint orchestrates the two stages; see run.sh -h for usage
COPY --chown=user:user run.sh ./run.sh
ENTRYPOINT ["/bin/bash", "/opt/app/run.sh"]
