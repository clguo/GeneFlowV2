from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .datasets import SAMPLES, check


SMOKE_MODEL = [
    '--img_size', '16', '--model_channels', '32', '--num_res_blocks', '1',
    '--channel_mult', '1', '2', '--attention_resolutions', '2',
    '--num_heads', '2', '--num_head_channels', '16',
    '--gene_token_dim', '32', '--gene_transformer_layers', '1',
    '--gene_transformer_heads', '4', '--gene_transformer_ff_dim', '64',
    '--cross_attention_resolutions', '1', '2', '--cross_attention_heads', '2',
    '--cross_attention_head_dim', '16', '--max_cross_attention_genes', '8',
    '--epochs', '1', '--batch_size', '4', '--num_workers', '0', '--save_every_epochs', '0',
]


def build_command(args, manifest, extra):
    prepared = args.prepared_dir or args.data_root / 'prepared' / args.dataset
    prepared = prepared.resolve()
    output = (args.output_dir or Path('outputs') / args.dataset).resolve()
    command = [
        sys.executable, '-m', f'geneflowv2.{args.action}',
        '--adata', str((prepared / manifest['adata']).resolve()),
        '--image_paths', str(prepared / manifest['image_paths']),
        '--split_manifest', str(prepared / manifest['split_manifest']),
    ]
    if args.action == 'train':
        command += [
            '--output_dir', str(output), '--img_channels', '4',
            '--spatial_key', manifest['spatial_key'],
            '--spatial_patch_size', str(manifest['spatial_patch_size']),
            '--seed', str(manifest['seed']), '--abort_on_nonfinite',
        ]
        if args.smoke:
            command += SMOKE_MODEL
        else:
            command += [
                '--img_size', '256', '--epochs', '80' if args.dataset == 'C2' else '50',
                '--batch_size', '8', '--num_workers', '4', '--patience', '5',
                '--use_checkpoint', '--use_amp', '--amp_dtype', 'bfloat16',
            ]
    else:
        checkpoint = args.checkpoint or output / 'checkpoints/best_checkpoint.pt'
        command += [
            '--model_path', str(checkpoint.resolve()),
            '--output_dir', str(output / 'validation'),
            '--batch_size', '4', '--num_workers', '0' if args.smoke else '4',
            '--num_samples', '4' if args.smoke else '64',
            '--gen_steps', '2' if args.smoke else '50', '--guidance_scale', '1.5',
        ]
    if extra and extra[0] == '--':
        extra = extra[1:]
    return command + extra


def main():
    parser = argparse.ArgumentParser(description='Train or validate GeneFlowV2 on prepared C1/C2/P1 data.')
    parser.add_argument('action', choices=['train', 'validate'])
    parser.add_argument('--dataset', required=True, choices=SAMPLES)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    prepared = args.prepared_dir or args.data_root / 'prepared' / args.dataset
    manifest = check(prepared, args.dataset)
    if args.smoke and (manifest['retained_cells'] > 64 or manifest['gene_count'] > 32):
        raise ValueError('For --smoke, prepare a separate view with --max-cells 32 --max-genes 16.')
    command = build_command(args, manifest, extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        env = dict(os.environ)
        if args.smoke:
            env['CUDA_VISIBLE_DEVICES'] = ''
            env.setdefault('OMP_NUM_THREADS', '2')
            env.setdefault('MKL_NUM_THREADS', '2')
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
