"""Unit tests for the final concat-latent post-processing stage.

These run with plain ``pytest`` — no Docker, GPU, slide2vec, or weights. They
exercise the pure on-disk function that turns the five per-fold ``[192]`` WSI
latents into the single ``[960]`` float32 deliverable latent per case.
"""

import sys
from pathlib import Path

import orjson
import torch

# make ``aggregator/`` importable so ``import postprocess`` works regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocess import concat_fold_latents  # noqa: E402

NUM_FOLDS = 5
FOLD_DIM = 192
CONCAT_DIM = NUM_FOLDS * FOLD_DIM  # 960


def _make_folds(folds_dir: Path, sample_ids, dtype=torch.float32):
    """Write synthetic per-fold ``[192]`` tensors and return them keyed by id+fold."""
    tensors = {}
    torch.manual_seed(0)
    for fold in range(NUM_FOLDS):
        fold_dir = folds_dir / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in sample_ids:
            t = torch.randn(FOLD_DIM, dtype=dtype)
            torch.save(t, fold_dir / f"{sample_id}.pt")
            tensors[(sample_id, fold)] = t
    return tensors


def test_concat_equals_manual_fold_order(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_id = "1100_4"
    tensors = _make_folds(folds_dir, [sample_id])

    written = concat_fold_latents(folds_dir, out_dir)

    out_path = out_dir / f"{sample_id}.pt"
    assert out_path in written
    result = torch.load(out_path)

    expected = torch.cat(
        [tensors[(sample_id, fold)].flatten() for fold in range(NUM_FOLDS)], dim=0
    ).to(torch.float32)

    assert torch.equal(result, expected)


def test_output_shape_and_dtype(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    _make_folds(folds_dir, ["caseA"])

    concat_fold_latents(folds_dir, out_dir)

    result = torch.load(out_dir / "caseA.pt")
    assert result.shape == (CONCAT_DIM,)
    assert result.dtype == torch.float32


def test_output_filename_carries_sample_id(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_ids = ["1100_4", "2200_1"]
    _make_folds(folds_dir, sample_ids)

    concat_fold_latents(folds_dir, out_dir)

    for sample_id in sample_ids:
        assert (out_dir / f"{sample_id}.pt").is_file()


def test_per_fold_inputs_left_in_place(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_id = "1100_4"
    _make_folds(folds_dir, [sample_id])

    concat_fold_latents(folds_dir, out_dir)

    for fold in range(NUM_FOLDS):
        assert (folds_dir / f"fold-{fold}" / f"{sample_id}.pt").is_file()


def test_casts_non_float32_inputs_to_float32(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_id = "dtype_case"
    tensors = _make_folds(folds_dir, [sample_id], dtype=torch.float64)

    concat_fold_latents(folds_dir, out_dir)

    result = torch.load(out_dir / f"{sample_id}.pt")
    assert result.dtype == torch.float32
    expected = torch.cat(
        [tensors[(sample_id, fold)].flatten() for fold in range(NUM_FOLDS)], dim=0
    ).to(torch.float32)
    assert torch.equal(result, expected)


def test_json_companion_written_for_each_case(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_ids = ["1100_4", "2200_1"]
    _make_folds(folds_dir, sample_ids)

    concat_fold_latents(folds_dir, out_dir)

    for sample_id in sample_ids:
        json_path = out_dir / f"{sample_id}.json"
        assert json_path.is_file()
        assert (out_dir / f"{sample_id}.pt").is_file()


def test_json_companion_is_grand_challenge_feature_wrapper(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_id = "1100_4"
    _make_folds(folds_dir, [sample_id])

    concat_fold_latents(folds_dir, out_dir)

    content = orjson.loads((out_dir / f"{sample_id}.json").read_bytes())

    assert isinstance(content, list)
    assert len(content) == 1
    element = content[0]
    assert element["title"] == ""
    assert len(element["features"]) == CONCAT_DIM


def test_json_features_roundtrip_to_pt_values(tmp_path):
    folds_dir = tmp_path / "latents" / "folds"
    out_dir = tmp_path / "latents"
    sample_id = "1100_4"
    _make_folds(folds_dir, [sample_id])

    concat_fold_latents(folds_dir, out_dir)

    content = orjson.loads((out_dir / f"{sample_id}.json").read_bytes())
    features = content[0]["features"]

    pt_values = torch.load(out_dir / f"{sample_id}.pt").tolist()

    # both derive from the same float32 tensor; the JSON path goes through
    # float(np.float32(x)), which is the same value tolist() yields -> exact.
    assert features == pt_values
