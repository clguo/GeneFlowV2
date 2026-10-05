#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUDA_VISIBLE_DEVICES=""
python scripts/make_demo_data.py
python -m geneflowv2.train \
  --adata demo_data/expression.h5ad \
  --image_paths demo_data/image_paths.json \
  --output_dir outputs/demo \
  --img_size 16 --img_channels 3 --spatial_patch_size 16 \
  --epochs 1 --batch_size 4 --num_workers 0 \
  --model_channels 32 --num_res_blocks 1 --channel_mult 1 2 \
  --attention_resolutions 2 --num_heads 2 --num_head_channels 16 \
  --gene_token_dim 32 --gene_transformer_layers 1 \
  --gene_transformer_heads 4 --gene_transformer_ff_dim 64 \
  --cross_attention_resolutions 1 2 \
  --cross_attention_heads 2 --cross_attention_head_dim 16 \
  --max_cross_attention_genes 8 --save_every_epochs 0 \
  --abort_on_nonfinite
python -m geneflowv2.validate \
  --adata demo_data/expression.h5ad \
  --image_paths demo_data/image_paths.json \
  --model_path outputs/demo/checkpoints/best_checkpoint.pt \
  --split_manifest outputs/demo/split_manifest.json \
  --output_dir outputs/demo/validation \
  --batch_size 4 --num_samples 4 --num_workers 0 \
  --gen_steps 2 --preview_samples 2
