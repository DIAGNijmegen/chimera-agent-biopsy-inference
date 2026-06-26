#!/usr/bin/env python
"""Final pipeline stage: concatenate the per-fold WSI latents into the deliverable.

The 5-fold HViT MIL aggregator writes one ``[192]`` WSI latent per case per fold
under ``latents/folds/fold-{0..4}/<sample_id>.pt``. The deliverable handed to the
external partner is ONE latent per case: the five flattened ``[192]`` fold latents
concatenated in fold order 0->4 into a single ``[960]`` float32 vector, written to
``latents/<sample_id>.pt``.

This module exposes a single pure function, :func:`concat_fold_latents`, that
operates purely on on-disk files (no Docker, GPU, slide2vec, or weights), and a
thin argparse CLI that delegates to it.
"""

import argparse
from pathlib import Path

import numpy as np
import orjson
import torch

NUM_FOLDS = 5


def sanitize_json_content(obj):
    """Recursively cast numpy scalars to plain Python types for JSON serialization.

    This mirrors the maintainer's notebook ``sanitize_json_content`` exactly so the
    emitted bytes match the already-submitted artifacts: numpy floats become
    ``float``, numpy ints become ``int``, and arrays/lists/tuples become lists.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json_content(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_json_content(v) for v in obj]
    elif isinstance(obj, (str, int, bool, float)):
        return obj
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(
        obj,
        (
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ),
    ):
        return int(obj)
    else:
        return obj.__repr__()


def write_json_file(*, location, features, title):
    """Write the grand-challenge feature-vector wrapper as orjson bytes.

    Builds ``[{"title": title, "features": features}]``, sanitizes numpy scalars to
    plain Python types, then writes ``orjson.dumps`` raw bytes. This replicates the
    maintainer's notebook serialization path so the JSON companion is byte-identical
    to the already-submitted artifacts.
    """
    output_dict = [{"title": title, "features": features}]
    content = sanitize_json_content(output_dict)
    with open(location, "wb") as f:
        f.write(orjson.dumps(content))


def concat_fold_latents(folds_dir, output_dir, num_folds: int = NUM_FOLDS):
    """Concatenate per-fold latents into one ``[960]`` float32 latent per case.

    For every case discovered under ``folds_dir/fold-0/`` (the canonical set of
    sample ids), the per-fold latents are loaded in fold order ``0..num_folds-1``,
    each flattened to 1-D, concatenated, cast to float32, and written to
    ``output_dir/<sample_id>.pt``. A language-agnostic JSON companion is also
    written to ``output_dir/<sample_id>.json`` (grand-challenge feature-vector
    wrapper). The per-fold inputs are left untouched.

    Args:
        folds_dir: directory containing ``fold-{0..N-1}/<sample_id>.pt``.
        output_dir: directory the concatenated ``<sample_id>.pt`` files go into.
        num_folds: number of folds to concatenate (default 5).

    Returns:
        Sorted list of the written ``Path`` objects.
    """
    folds_dir = Path(folds_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold0_dir = folds_dir / "fold-0"
    if not fold0_dir.is_dir():
        raise FileNotFoundError(f"missing fold directory: {fold0_dir}")

    sample_ids = sorted(p.stem for p in fold0_dir.glob("*.pt"))

    written = []
    for sample_id in sample_ids:
        parts = []
        for fold in range(num_folds):
            fold_path = folds_dir / f"fold-{fold}" / f"{sample_id}.pt"
            if not fold_path.is_file():
                raise FileNotFoundError(
                    f"missing per-fold latent for case '{sample_id}': {fold_path}"
                )
            parts.append(torch.load(fold_path, map_location="cpu").flatten())

        latent = torch.cat(parts, dim=0).to(torch.float32)

        out_path = output_dir / f"{sample_id}.pt"
        torch.save(latent, out_path)
        written.append(out_path)

        # Language-agnostic JSON companion next to the .pt. Replicate the
        # notebook path exactly: [960] tensor -> np.array(...) -> sanitize ->
        # orjson.dumps, so the bytes match the already-submitted artifacts.
        write_json_file(
            location=out_path.with_suffix(".json"),
            features=np.array(latent),
            title="",
        )

    return sorted(written)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        "Concatenate per-fold WSI latents into the deliverable latent",
        add_help=add_help,
    )
    parser.add_argument(
        "--latents-dir",
        required=True,
        type=str,
        help="latents directory; per-fold inputs live under '<latents-dir>/folds/' "
        "and the concatenated '<sample_id>.pt' files are written into '<latents-dir>/'",
    )
    parser.add_argument(
        "--num-folds",
        default=NUM_FOLDS,
        type=int,
        help="number of folds to concatenate (default: 5)",
    )
    return parser


def main(args):
    latents_dir = Path(args.latents_dir)
    folds_dir = latents_dir / "folds"
    written = concat_fold_latents(folds_dir, latents_dir, num_folds=args.num_folds)
    print(f"Wrote {len(written)} concatenated latent(s) to {latents_dir}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    raise SystemExit(main(args))
