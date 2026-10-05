from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

import anndata as ad
import numpy as np
import tifffile
from scipy import sparse

from .spatial_split import create_spatial_split_manifest, load_split_manifest, save_split_manifest


SAMPLES = {
    'C1': ('Xenium_V1_hSkin_Melanoma_Base_FFPE', '04ae674e0ba5ade2c8c55e0591c77b26'),
    'C2': ('Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE', 'c013c4e3bad8a960fd37628c14ef7148'),
    'P1': ('Xenium_Prime_Human_Skin_FFPE', 'c1750fb960b6f3380fc58cfbaa1a6ce3'),
}
RECORD_URL = 'https://zenodo.org/records/17429142'
INPUT_DIR = Path('cell_patch_256_aux/input')


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def extract_archive(archive_path, destination_root, dataset):
    sample_name = SAMPLES[dataset][0]
    destination_root = Path(destination_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / sample_name
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; use prepare for extracted data.')
    staging = Path(tempfile.mkdtemp(prefix=f'.{dataset}-extract-', dir=destination_root))
    try:
        count = 0
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if member.name.startswith('/') or '..' in parts:
                    raise ValueError(f'Unsafe archive path: {member.name}')
                if parts and parts[0] == 'processed_data':
                    parts = parts[1:]
                if not parts or parts[0] != sample_name:
                    if member.isdir() and not parts:
                        continue
                    raise ValueError(f'Unexpected dataset in archive: {member.name}')
                relative = Path(*parts[1:])
                needed = (
                    relative == Path('adata.h5ad')
                    or relative == INPUT_DIR / 'cell_image_paths.json'
                    or (INPUT_DIR / 'cell_images') in relative.parents
                )
                if not needed or member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f'Unsupported archive member: {member.name}')
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                count += 1
                if count % 10000 == 0:
                    print(f'Extracted {count} single-cell files', flush=True)
        for relative in ['adata.h5ad', INPUT_DIR / 'cell_image_paths.json', INPUT_DIR / 'cell_images']:
            if not (staging / relative).exists():
                raise FileNotFoundError(f'Archive is missing {relative}')
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination


def download(dataset, data_root):
    data_root = Path(data_root).resolve()
    name, checksum = SAMPLES[dataset]
    destination = data_root / 'raw' / name
    if destination.exists():
        print(f'Using extracted sample: {destination}')
        return destination
    downloads = data_root / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f'{name}.tar.gz'
    if not archive.exists():
        if shutil.which('curl') is None:
            raise RuntimeError('Install curl, or download the archive manually and use --archive.')
        partial = archive.with_suffix('.gz.part')
        subprocess.run([
            'curl', '--fail', '--location', '--retry', '5', '--continue-at', '-',
            '--output', str(partial), f'{RECORD_URL}/files/{name}.tar.gz?download=1',
        ], check=True)
        if digest(partial, 'md5') != checksum:
            raise ValueError(f'Archive MD5 mismatch: {partial}. Remove it and retry the download.')
        partial.rename(archive)
    elif digest(archive, 'md5') != checksum:
        raise ValueError(f'Archive MD5 mismatch: {archive}')
    return extract_archive(archive, data_root / 'raw', dataset)


def prepare(dataset, sample_dir, output_dir, seed=42, val_fraction=0.2,
            patch_size=256, spatial_key='spatial', max_cells=None, max_genes=None):
    sample_dir = Path(sample_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} already exists; use check or choose another --output-dir.')
    if sample_dir.name != SAMPLES[dataset][0]:
        raise ValueError(f'{dataset} expects the sample directory {SAMPLES[dataset][0]}')
    if max_cells is not None and max_cells < 4:
        raise ValueError('--max-cells must be at least 4')
    if max_genes is not None and max_genes < 1:
        raise ValueError('--max-genes must be positive')
    adata_path = sample_dir / 'adata.h5ad'
    image_dir = sample_dir / INPUT_DIR / 'cell_images'
    index_path = sample_dir / INPUT_DIR / 'cell_image_paths.json'
    released = json.loads(index_path.read_text())
    if not isinstance(released, dict) or not released:
        raise ValueError(f'Expected a non-empty image mapping: {index_path}')
    local_names = {entry.name for entry in os.scandir(image_dir) if entry.is_file()}
    source = ad.read_h5ad(adata_path, backed='r')
    staging = None
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError('Cell IDs and gene names must be unique')
        if spatial_key not in source.obsm:
            raise KeyError(f'Missing AnnData obsm[{spatial_key!r}]')
        ids = [str(x) for x in source.obs_names]
        xy = np.asarray(source.obsm[spatial_key])
        if xy.ndim != 2 or xy.shape[0] != len(ids) or xy.shape[1] < 2:
            raise ValueError(f'Invalid spatial coordinates: {xy.shape}')
        local = {}
        positions = {}
        for row, cell_id in enumerate(ids):
            if cell_id not in released or not np.isfinite(xy[row, :2]).all():
                continue
            basename = str(released[cell_id]).replace('\\', '/').rsplit('/', 1)[-1]
            candidates = (basename, f'{cell_id}_original.tif')
            match = next((n for n in candidates if n in local_names), None)
            if match is not None:
                local[cell_id] = image_dir / match
                positions[cell_id] = row
        available = len(local)
        if max_cells is not None and available > max_cells:
            selected = np.random.default_rng(seed).choice(sorted(local), max_cells, replace=False)
            local = {cell_id: local[cell_id] for cell_id in sorted(selected)}
        if len(local) < 4:
            raise ValueError('Need at least four cells with local images and finite coordinates')
        coordinates = {cell_id: xy[positions[cell_id], :2] for cell_id in local}
        split = create_spatial_split_manifest(
            sorted(local), coordinates, val_fraction=val_fraction,
            patch_size_pixels=patch_size, seed=seed, spatial_key=spatial_key,
        )
        if min(split['train_count'], split['val_count']) < 2:
            raise ValueError('Spatial split needs at least two train and two validation cells')
        ngenes = min(source.n_vars, max_genes or source.n_vars)
        gene_names = [str(x) for x in source.var_names[:ngenes]]
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f'.{dataset}-prepare-', dir=output_dir.parent))
        expression_path = adata_path
        if max_cells is not None or max_genes is not None:
            rows = sorted(positions[x] for x in local)
            matrix = source.X[rows, :][:, :ngenes]
            subset = ad.AnnData(X=matrix, obs=source.obs.iloc[rows].copy(), var=source.var.iloc[:ngenes].copy())
            subset.obsm[spatial_key] = xy[rows]
            subset.write_h5ad(staging / 'expression.h5ad')
            expression_path = output_dir / 'expression.h5ad'
        paths = {key: os.path.relpath(local[key], output_dir) for key in sorted(local)}
        write_json(staging / 'cell_image_paths.json', paths)
        (staging / 'gene_symbols.txt').write_text('\n'.join(gene_names) + '\n')
        save_split_manifest(split, staging / 'split_manifest.json')
        artifacts = ['cell_image_paths.json', 'gene_symbols.txt', 'split_manifest.json']
        if (staging / 'expression.h5ad').exists():
            artifacts.append('expression.h5ad')
        manifest = {
            'dataset': dataset, 'sample_name': sample_dir.name, 'source_url': RECORD_URL,
            'adata': os.path.relpath(expression_path, output_dir),
            'image_paths': 'cell_image_paths.json', 'split_manifest': 'split_manifest.json',
            'gene_symbols': 'gene_symbols.txt', 'gene_count': ngenes,
            'source_cells': len(ids), 'released_index_entries': len(released),
            'available_cells': available, 'retained_cells': len(local),
            'unavailable_cells': len(ids) - available,
            'subset': max_cells is not None or max_genes is not None,
            'seed': seed, 'spatial_key': spatial_key, 'spatial_patch_size': patch_size,
            'train_count': split['train_count'], 'val_count': split['val_count'],
            'buffer_dropped_count': split['buffer_dropped_count'],
            'overlap_audit': split['overlap_audit_train_cells_with_val_overlap'],
            'artifact_sha256': {name: digest(staging / name) for name in artifacts},
        }
        write_json(staging / 'dataset.json', manifest)
        os.replace(staging, output_dir)
        staging = None
    finally:
        source.file.close()
        if staging is not None:
            shutil.rmtree(staging)
    print(json.dumps(manifest, indent=2))
    return output_dir


def check(prepared_dir, dataset, image_checks=8):
    prepared_dir = Path(prepared_dir).resolve()
    manifest = json.loads((prepared_dir / 'dataset.json').read_text())
    if manifest['dataset'] != dataset:
        raise ValueError(f'Prepared data belong to {manifest["dataset"]}, not {dataset}')
    for name, expected in manifest['artifact_sha256'].items():
        if digest(prepared_dir / name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}; prepare into a new directory')
    paths = json.loads((prepared_dir / manifest['image_paths']).read_text())
    for cell_id, relative in paths.items():
        if not (prepared_dir / relative).is_file():
            raise FileNotFoundError(f'Image missing for {cell_id}: {relative}')
    split = load_split_manifest(prepared_dir / manifest['split_manifest'], paths)
    source = ad.read_h5ad(prepared_dir / manifest['adata'], backed='r')
    try:
        genes = (prepared_dir / manifest['gene_symbols']).read_text().splitlines()
        if list(source.var_names.astype(str)) != genes:
            raise ValueError('Prepared gene order differs from AnnData')
        if not set(paths).issubset(set(source.obs_names)):
            raise ValueError('Prepared cell IDs are missing from AnnData')
        count = len(paths) if image_checks < 0 else min(image_checks, len(paths))
        selected = np.random.default_rng(manifest['seed']).choice(sorted(paths), count, replace=False)
        positions = sorted(source.obs_names.get_indexer(selected))
        matrix = source.X[positions, :]
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite expression values')
        for cell_id in selected:
            image = tifffile.imread(prepared_dir / paths[cell_id])
            if image.ndim != 3 or image.shape[-1] < 4 or not np.isfinite(image).all():
                raise ValueError(f'{cell_id}: expected a finite HWC image with at least four channels, got {image.shape}')
        print(f'{dataset}: {len(paths)} cells, {len(genes)} genes, '
              f'{split["train_count"]} train / {split["val_count"]} val; '
              f'{count} images decoded; overlap audit={manifest["overlap_audit"]}', flush=True)
    finally:
        source.file.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Download, prepare, and check released C1/C2/P1 data.')
    parser.add_argument('action', choices=['download', 'prepare', 'check'])
    parser.add_argument('--dataset', required=True, choices=SAMPLES)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--sample-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--patch-size', type=float, default=256)
    parser.add_argument('--spatial-key', default='spatial')
    parser.add_argument('--max-cells', type=int)
    parser.add_argument('--max-genes', type=int)
    parser.add_argument('--image-checks', type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir or args.data_root / 'prepared' / args.dataset
    if args.action == 'check':
        check(output, args.dataset, args.image_checks)
        return
    sample = args.sample_dir or args.data_root / 'raw' / SAMPLES[args.dataset][0]
    if args.action == 'download':
        if args.archive:
            if digest(args.archive, 'md5') != SAMPLES[args.dataset][1]:
                raise ValueError(f'Archive MD5 mismatch: {args.archive}')
            sample = extract_archive(args.archive, args.data_root / 'raw', args.dataset)
        else:
            sample = download(args.dataset, args.data_root)
    if output.exists():
        raise FileExistsError(f'{output} exists; use check to verify it or choose another --output-dir')
    prepare(args.dataset, sample, output, args.seed, args.val_fraction,
            args.patch_size, args.spatial_key, args.max_cells, args.max_genes)
    check(output, args.dataset, args.image_checks)


if __name__ == '__main__':
    main()
