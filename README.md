# GeneFlowV2

A standalone PyTorch implementation of GeneFlowV2 for RNA-conditioned cell image generation. This repository contains the V2 architecture, training, and held-out conditioning validation. Shared neural-network components are included locally; no earlier GeneFlow repository or checkpoint is required.

## Architecture

Gene identity and expression values form tokens processed by a Transformer. All observed genes, including measured zeros, contribute to a pooled cell embedding. The embedding conditions a UNet through block-specific FiLM adapters. A cell-summary token and expression-selected gene tokens supply multi-scale spatial cross-attention. Genes absent from the assay are masked separately from measured zeros.

Training combines a sinusoidal rectified-flow objective with a low-time correct-versus-shuffled RNA ranking objective. Flow gradients update the full model; additional ranking gradients update only the RNA encoder, FiLM adapters, and spatial cross-attention path. Training includes condition dropout for classifier-free guidance, EMA weights, checkpoint resume, and a spatially separated validation split with a crop-overlap guard band.

Default routing is `hard` (top expressed genes). `--routing_mode soft` enables continuous expression weighting; `all_observed` is also supported. These modes keep the same parameter layout but change conditioning behavior.

## Quick start

Run commands from the repository root. Use Python 3.11 and a virtual environment. The release was checked with Python 3.11, PyTorch 2.2.2, and torchvision 0.17.2 in the existing development environment. The synthetic example runs on CPU and does not download a dataset or pretrained weights.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/quick_start.sh
```

The script creates 24 synthetic cells, trains a reduced 16 × 16 RGB model for one epoch, and validates its EMA checkpoint. It deliberately forces CPU execution. This checks the complete pipeline; its generated images and metrics are not research results.

Expected outputs:

```text
outputs/demo/
├── checkpoints/best_checkpoint.pt
├── checkpoints/latest_checkpoint.pt
├── gene_names.json
├── geneflowv2_config.json
├── split_manifest.json
├── training_history.csv
└── validation/
    ├── geneflowv2_conditioning_metrics.csv
    ├── geneflowv2_conditioning_summary.json
    ├── geneflowv2_gene_attention.csv
    └── geneflowv2_conditioning_preview.png
```

Re-running the demo writes to the same output directory. For a separate experiment, change `--output_dir` in the script or use the commands below.

## Prepare your data

Provide an AnnData file and a JSON mapping from cell IDs to image files:

```text
data/
├── expression.h5ad
├── image_paths.json
└── images/
    ├── cell_001.tif
    └── cell_002.tif
```

```json
{
  "cell_001": "images/cell_001.tif",
  "cell_002": "images/cell_002.tif"
}
```

Image paths may be absolute or relative to the JSON file's directory. Every listed file must exist. Only cells present in both AnnData and the mapping are used.

- `adata.X`: finite expression values, with cells in rows and genes in columns. `--layer counts` selects a named layer instead. Expression values are used as provided: no automatic library-size normalization or log transform is applied. Use the same preprocessing for training and validation; avoid gene-wise centering if zeros should retain their measured-zero meaning.
- `adata.obs_names`: unique cell IDs matching the JSON keys.
- `adata.var_names`: unique gene names. Validation checks the gene set and restores the checkpoint's gene order.
- `adata.obsm["spatial"]`: cell-center coordinates of shape `[n_cells, 2]`, in the original image's pixel coordinate system. Use `--spatial_key` for another key.
- Images: channel-last TIFF arrays `[H, W, C]`, or HDF5 files with an `image` dataset of that shape. Use `--img_channels 3` for RGB or `4` for RGB plus one auxiliary channel. The first requested channels are read. Non-uint8 images receive per-image min-max conversion to uint8; uint8 values are scaled to `[0, 1]`. Images are resized to `--img_size`. Invalid images raise an error.

`--spatial_patch_size` must describe the original crop width in the coordinate system above, even if the model resizes images. The split assumes axis-aligned square crops from one coordinate system. Train separate runs for independent slides unless you supply a correctly prepared split manifest. A new split removes training crops overlapping validation crops; the requested fraction is approximate. An existing manifest is reused and its cell IDs/checksum are checked.

Optional `--gene_symbols genes.csv` defines an ordered panel (one gene per line, no header); genes absent from AnnData are padded with zero and masked. `--missing_gene_symbols missing.csv` explicitly masks additional unmeasured genes. Apply the same panel and masking options during validation. Optional cell-type and count filters are listed in `--help`.

## Train

A full-size RGB run on one GPU:

```bash
python -m geneflowv2.train \
  --adata data/expression.h5ad \
  --image_paths data/image_paths.json \
  --output_dir outputs/run1 \
  --img_size 256 --img_channels 3 \
  --spatial_patch_size 256 --val_fraction 0.2 \
  --epochs 50 --batch_size 4 --num_workers 4 \
  --use_checkpoint --use_amp --amp_dtype bfloat16 \
  --abort_on_nonfinite
```

The Python entry point selects CUDA when available, otherwise CPU. Full-size training is intended for a GPU; reduce the batch size if needed, keeping it at least 2 for ranking. Use `--amp_dtype float16` if bfloat16 is unsupported, or omit `--use_amp` for full precision. The implementation uses one process and one device.

The default model has 128 base UNet channels, channel multipliers `[1, 2, 2, 2, 2]`, a 3-layer gene Transformer with 256-dimensional tokens, and up to 128 routed gene tokens. Attention resolution arguments are UNet downsampling factors, not pixel dimensions. Model channels must be compatible with 32-group normalization, and image dimensions with the UNet downsampling depth.

The validation loader uses EMA weights. The best checkpoint minimizes `main_loss + selection_ranking_weight * ranking_loss` (default weight: 5). The optimizer uses `condition_ranking_weight` (default: 0.5) for the additional conditioning gradients. A final singleton batch is dropped because RNA shuffling requires at least two cells.

To resume, repeat the same training command with `--auto_resume`, or add:

```text
--resume_from outputs/run1/checkpoints/latest_checkpoint.pt
```

Use the same data, gene order, model settings, training-loss settings, and split. `--epochs` is the total target epoch count. Early-stopped runs also require a larger `--patience` to continue. `--save_every_steps N` enables intermediate checkpoints. For reproducible experiments, keep the batch size, workers, seeds, and software environment fixed.

## Validate

Use the same input data and preprocessing as training:

```bash
python -m geneflowv2.validate \
  --adata data/expression.h5ad \
  --image_paths data/image_paths.json \
  --model_path outputs/run1/checkpoints/best_checkpoint.pt \
  --split_manifest outputs/run1/split_manifest.json \
  --output_dir outputs/run1/validation \
  --batch_size 4 --num_samples 64 --num_workers 4 \
  --gen_steps 50 --guidance_scale 1.5
```

Validation loads EMA weights, verifies the split checksum, and uses held-out cells only. Image/model dimensions are read from the checkpoint. Pass `--num_samples -1` for the entire validation split; a final singleton batch is dropped. Provide `--split_manifest` explicitly when moving a training output to another machine.

Reported quantities include correct, shuffled, and null-condition velocity MSE; low-time shuffled-minus-correct MSE; the fraction where correct RNA beats shuffled RNA; generated-image changes under RNA shuffling and a changed noise seed; routed gene counts; and gene attention. Positive shuffled-minus-correct MSE indicates an advantage for correct RNA. These are conditioning diagnostics, not FID, LPIPS, biological correctness, or causal validation.

Generation uses fixed-step Euler integration with classifier-free guidance. The preview columns are: real image, correct RNA, shuffled RNA, null condition, alternate noise, and normalized correct-versus-shuffled difference. Previews show the first three channels. Attention values are diagnostics and are not causal gene attributions.

## Source layout

```text
geneflowv2/
├── model.py           # GeneFlowV2 and expression routing
├── blocks.py          # Transformer, FiLM UNet, attention, and shared blocks
├── train.py           # Training, EMA, checkpoints, and epoch validation
├── validate.py        # Held-out conditioning metrics and previews
├── flow.py            # Flow path, sampling, EMA, and data-path helpers
├── data.py            # AnnData and image loading
└── spatial_split.py   # Spatial split creation and checks
scripts/
├── make_demo_data.py
└── quick_start.sh
tests/
└── test_model.py
```

This release uses the name GeneFlowV2 throughout the Python API, commands, output files, and checkpoint method metadata. Its architecture and training objectives are described above; checkpoints must match both the method and architecture identifiers. The package removes dependencies on older model/training entry points and removes experiment-specific ablation matching. It resolves relative image paths against the JSON directory and fails on invalid images. Two summary fields are named `uniform_correct_better_fraction` and `low_t_correct_better_fraction` to describe their actual sign convention.

Run architecture and conditioning-gradient checks:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
python -m geneflowv2.train --help
python -m geneflowv2.validate --help
```

The release includes source and a synthetic-data generator. Real datasets, trained weights, cluster scripts, logs, and experiment outputs are not included. The root MIT license and its original copyright notice are retained in [LICENSE](LICENSE).
