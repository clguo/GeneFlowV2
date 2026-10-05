from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


SPLIT_VERSION = 1


def _canonical_ids(cell_ids: Iterable[str]) -> list[str]:
    return sorted({str(cell_id) for cell_id in cell_ids})


def cell_id_sha256(cell_ids: Iterable[str]) -> str:
    payload = "\n".join(_canonical_ids(cell_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_spatial_coordinates(
    adata_path: str | Path,
    cell_ids: Sequence[str],
    spatial_key: str = "spatial",
) -> Tuple[Dict[str, Tuple[float, float]], list[str]]:

    import anndata as ad

    adata = ad.read_h5ad(str(adata_path), backed="r")
    try:
        if spatial_key not in adata.obsm:
            available = sorted(str(key) for key in adata.obsm.keys())
            raise KeyError(
                f"AnnData obsm does not contain {spatial_key!r}; available keys: {available}"
            )
        obs_names = [str(cell_id) for cell_id in adata.obs_names]
        coordinates = np.asarray(adata.obsm[spatial_key])
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have shape (n_cells, >=2), "
            f"got {coordinates.shape}"
        )
    if coordinates.shape[0] != len(obs_names):
        raise ValueError(
            f"Coordinate rows ({coordinates.shape[0]}) do not match obs_names ({len(obs_names)})"
        )

    requested = set(str(cell_id) for cell_id in cell_ids)
    result: Dict[str, Tuple[float, float]] = {}
    invalid = []
    for cell_id, xy in zip(obs_names, coordinates[:, :2]):
        if cell_id not in requested:
            continue
        x, y = float(xy[0]), float(xy[1])
        if math.isfinite(x) and math.isfinite(y):
            result[cell_id] = (x, y)
        else:
            invalid.append(cell_id)

    missing = sorted(requested.difference(result).difference(invalid))
    return result, sorted(invalid + missing)


def _split_at_val_count(
    order: np.ndarray,
    coordinates: np.ndarray,
    val_count: int,
    patch_size_pixels: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    val_indices = order[-val_count:]
    train_candidates = order[:-val_count]
    if train_candidates.size == 0:
        return train_candidates, val_indices, train_candidates, 1.0


    overlap_radius = np.nextafter(float(patch_size_pixels), 0.0)
    tree = cKDTree(coordinates[val_indices])
    overlap_counts = tree.query_ball_point(
        coordinates[train_candidates],
        r=overlap_radius,
        p=np.inf,
        workers=-1,
        return_length=True,
    )
    overlaps = np.asarray(overlap_counts) > 0
    train_indices = train_candidates[~overlaps]
    buffer_indices = train_candidates[overlaps]
    retained = train_indices.size + val_indices.size
    actual_val_fraction = val_indices.size / retained if retained else 1.0
    return train_indices, val_indices, buffer_indices, actual_val_fraction


def create_spatial_split_manifest(
    cell_ids: Sequence[str],
    coordinates_by_id: Mapping[str, Sequence[float]],
    val_fraction: float = 0.2,
    patch_size_pixels: float = 256.0,
    seed: int = 42,
    spatial_key: str = "spatial",
) -> Dict[str, Any]:


    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if patch_size_pixels <= 0:
        raise ValueError(f"patch_size_pixels must be positive, got {patch_size_pixels}")

    requested_ids = _canonical_ids(cell_ids)
    valid_ids = [cell_id for cell_id in requested_ids if cell_id in coordinates_by_id]
    missing_ids = sorted(set(requested_ids).difference(valid_ids))
    if len(valid_ids) < 2:
        raise ValueError("At least two cells with finite spatial coordinates are required")

    coordinates = np.asarray(
        [coordinates_by_id[cell_id][:2] for cell_id in valid_ids], dtype=np.float64
    )
    if coordinates.shape != (len(valid_ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError("All supplied spatial coordinates must be finite two-dimensional values")

    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, math.pi))
    direction = np.asarray([math.cos(angle), math.sin(angle)])
    projection = coordinates @ direction

    order = np.lexsort((np.arange(len(valid_ids)), projection))

    low, high = 1, len(valid_ids) - 1
    best = None
    while low <= high:
        val_count = (low + high) // 2
        candidate = _split_at_val_count(
            order, coordinates, val_count, patch_size_pixels
        )
        error = abs(candidate[3] - val_fraction)
        if best is None or error < best[0]:
            best = (error, val_count, candidate)
        if candidate[3] < val_fraction:
            low = val_count + 1
        else:
            high = val_count - 1

    assert best is not None
    _, selected_val_count, (train_idx, val_idx, buffer_idx, actual_fraction) = best
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError(
            "Spatial split left an empty train or validation set. Reduce the patch-size "
            "guard band or use a larger spatial field."
        )


    audit_tree = cKDTree(coordinates[val_idx])
    audit_counts = audit_tree.query_ball_point(
        coordinates[train_idx],
        r=np.nextafter(float(patch_size_pixels), 0.0),
        p=np.inf,
        workers=-1,
        return_length=True,
    )
    overlap_pairs_detected = int(np.count_nonzero(audit_counts))
    if overlap_pairs_detected:
        raise AssertionError(
            f"Spatial split audit found {overlap_pairs_detected} overlapping training cells"
        )

    manifest: Dict[str, Any] = {
        "version": SPLIT_VERSION,
        "strategy": "spatial_contiguous_holdout_with_guard_band",
        "seed": int(seed),
        "spatial_key": str(spatial_key),
        "coordinate_units": "pixels",
        "patch_size_pixels": float(patch_size_pixels),
        "val_fraction_requested": float(val_fraction),
        "val_fraction_retained": float(actual_fraction),
        "projection_angle_radians": angle,
        "projection_tail": "high",
        "selected_val_candidates": int(selected_val_count),
        "dataset_cell_count": len(requested_ids),
        "coordinate_cell_count": len(valid_ids),
        "retained_cell_count": int(train_idx.size + val_idx.size),
        "train_count": int(train_idx.size),
        "val_count": int(val_idx.size),
        "buffer_dropped_count": int(buffer_idx.size),
        "missing_coordinate_count": len(missing_ids),
        "overlap_audit_train_cells_with_val_overlap": overlap_pairs_detected,
        "overlap_rule": (
            "For 256-style cell-centred axis-aligned crops, train/val overlap is "
            "forbidden when abs(dx) < patch_size_pixels and abs(dy) < patch_size_pixels."
        ),
        "dataset_cell_id_sha256": cell_id_sha256(requested_ids),
        "train_cell_ids": sorted(valid_ids[index] for index in train_idx),
        "val_cell_ids": sorted(valid_ids[index] for index in val_idx),
        "buffer_dropped_cell_ids": sorted(valid_ids[index] for index in buffer_idx),
        "missing_coordinate_cell_ids": missing_ids,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def save_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(dict(manifest), handle, indent=2)
    return destination


def load_split_manifest(
    path: str | Path,
    dataset_cell_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    source = Path(path)
    with source.open() as handle:
        manifest = json.load(handle)
    if manifest.get("version") != SPLIT_VERSION:
        raise ValueError(
            f"Unsupported split manifest version {manifest.get('version')}; "
            f"expected {SPLIT_VERSION}"
        )
    expected_hash = manifest.get("manifest_sha256")
    actual_hash = manifest_sha256(manifest)
    if expected_hash != actual_hash:
        raise ValueError(f"Split manifest checksum mismatch: {source}")

    train_ids = set(str(cell_id) for cell_id in manifest.get("train_cell_ids", []))
    val_ids = set(str(cell_id) for cell_id in manifest.get("val_cell_ids", []))
    if not train_ids or not val_ids:
        raise ValueError(f"Split manifest has an empty train or validation set: {source}")
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise ValueError(f"Split manifest has train/validation ID overlap: {sorted(overlap)[:10]}")

    if dataset_cell_ids is not None:
        dataset_ids = set(str(cell_id) for cell_id in dataset_cell_ids)
        unavailable = (train_ids | val_ids).difference(dataset_ids)
        if unavailable:
            raise ValueError(
                f"Split manifest contains {len(unavailable)} cells absent from the current dataset; "
                f"examples: {sorted(unavailable)[:10]}"
            )
        expected_dataset_hash = manifest.get("dataset_cell_id_sha256")
        current_dataset_hash = cell_id_sha256(dataset_cell_ids)
        if expected_dataset_hash != current_dataset_hash:
            raise ValueError(
                "Current dataset cell IDs do not match the dataset used to create the split manifest"
            )
    return manifest


def dataset_view(dataset: Any, cell_ids: Sequence[str]) -> Any:

    available = set(str(cell_id) for cell_id in dataset.cell_ids)
    requested = [str(cell_id) for cell_id in cell_ids]
    missing = [cell_id for cell_id in requested if cell_id not in available]
    if missing:
        raise ValueError(f"Dataset view requested unavailable cells: {missing[:10]}")
    view = copy.copy(dataset)
    view.cell_ids = requested
    return view


def split_dataset_views(dataset: Any, manifest: Mapping[str, Any]) -> Tuple[Any, Any]:
    return (
        dataset_view(dataset, manifest["train_cell_ids"]),
        dataset_view(dataset, manifest["val_cell_ids"]),
    )


def default_manifest_for_model(model_path: str | Path) -> Path:

    model_path = Path(model_path)
    if model_path.parent.name == "checkpoints":
        return model_path.parent.parent / "split_manifest.json"
    return model_path.parent / "split_manifest.json"
