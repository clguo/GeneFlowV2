# GeneFlowV2

RNA-conditioned cell image generation with expression-routed attention and condition-only ranking. This standalone repository includes the architecture, **C1/C2/P1 dataset download and loading**, training, and held-out validation. The complete example below uses **C1**.

## Install

Run commands from the repository root on Linux or macOS with Python 3.11. Full training uses one CUDA GPU; the CPU checks below use a reduced model. `curl` is needed only for automatic downloads.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The pinned dependencies were tested in a Python 3.11 environment with PyTorch 2.2.2 and torchvision 0.17.2. Full datasets are much larger than the CPU examples: allow space for both the downloaded archive and extracted cell images. The training loader materializes expression data in memory; P1 also has a substantially larger gene panel and higher attention-memory requirements than C1/C2.

## Quick start: C1

### 1. Download and prepare C1

```bash
python -m geneflowv2.datasets download --dataset C1 --data-root data
```

This downloads the [published C1 archive](https://zenodo.org/records/17429142/files/Xenium_V1_hSkin_Melanoma_Base_FFPE.tar.gz?download=1), verifies its published MD5, extracts AnnData and single-cell images, repairs the released image paths, intersects image IDs with AnnData, and creates an approximately 80:20 spatial split with a crop-overlap guard band. The command checks all retained image paths and decodes eight sample images. Patch-level images are not needed and are skipped during extraction.

The download can be resumed by rerunning the command after a network interruption. Once preparation is complete, use `check` instead of preparing over an existing directory:

```bash
python -m geneflowv2.datasets check --dataset C1 --data-root data
```

Expected layout:

```text
data/
├── downloads/Xenium_V1_hSkin_Melanoma_Base_FFPE.tar.gz
├── raw/Xenium_V1_hSkin_Melanoma_Base_FFPE/
│   ├── adata.h5ad
│   └── cell_patch_256_aux/input/
│       ├── cell_image_paths.json
│       └── cell_images/
└── prepared/C1/
    ├── dataset.json
    ├── cell_image_paths.json
    ├── gene_symbols.txt
    └── split_manifest.json
```

`data/raw/.../cell_image_paths.json` is the original released index and may contain invalid absolute paths. Training uses **`data/prepared/C1/cell_image_paths.json`**, whose paths are relative to that file. The original data are not rewritten or copied during full preparation. Move the complete `data/` tree together to preserve these relative paths.

### 2. Train C1

```bash
python -m geneflowv2.run train --dataset C1 --data-root data
```

The launcher resolves the AnnData, repaired image index, and spatial split from `dataset.json`. Defaults: 256 × 256 images, the first four TIFF channels, batch size 8, 50 epochs, EMA, bfloat16 mixed precision, and gradient checkpointing. Outputs go to `outputs/C1/`. Reduce the batch size to 2 if needed, and use float16 if your GPU does not support bfloat16:

```bash
python -m geneflowv2.run train --dataset C1 --data-root data \
  -- --batch_size 2 --amp_dtype float16
```

Arguments after `--` are forwarded to the underlying training/validation entry point and override its defaults. To inspect the command without starting training:

```bash
python -m geneflowv2.run train --dataset C1 --data-root data --dry-run
```

### 3. Validate C1

```bash
python -m geneflowv2.run validate --dataset C1 --data-root data
```

Validation loads `outputs/C1/checkpoints/best_checkpoint.pt`, checks its split checksum and gene order, and evaluates held-out cells using EMA weights. It uses 64 cells, 50 Euler steps, and guidance scale 1.5 by default. To evaluate all validation cells:

```bash
python -m geneflowv2.run validate --dataset C1 --data-root data -- --num_samples -1
```

Outputs:

```text
outputs/C1/
├── checkpoints/best_checkpoint.pt
├── checkpoints/latest_checkpoint.pt
├── gene_names.json
├── geneflowv2_config.json
├── training_history.csv
└── validation/
    ├── geneflowv2_conditioning_metrics.csv
    ├── geneflowv2_conditioning_summary.json
    ├── geneflowv2_gene_attention.csv
    └── geneflowv2_conditioning_preview.png
```

The spatial manifest remains in `data/prepared/C1/`. Keep it with the checkpoint. `--output-dir outputs/my_run` changes the run directory; pass it to both train and validate. `--checkpoint path/to/checkpoint.pt` selects another validation checkpoint.

To execute download/preparation, training, and validation in sequence:

```bash
bash scripts/quick_start.sh C1
```

This is a **full C1 experiment**, not the synthetic CPU check. It reuses prepared data and resumes an existing latest checkpoint. `DATA_ROOT` and `OUTPUT_DIR` can override its directories.

## C2 and P1

The same loader explicitly supports each released sample and checks the requested dataset identity. Each sample keeps its own cells, gene panel, spatial split, and checkpoint; panels are not pooled or forced into C1's gene order.

| Dataset | Released sample directory | Gene count in checked data | Compressed archive |
|---|---|---:|---:|
| C1 | `Xenium_V1_hSkin_Melanoma_Base_FFPE` | 282 | 32.0 GB |
| C2 | `Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE` | 382 | 26.0 GB |
| P1 | `Xenium_Prime_Human_Skin_FFPE` | 5006 | 56.3 GB |

Archive names, sizes, and checksums come from the [GeneFlow dataset release](https://zenodo.org/records/17429142). C2 uses 80 training epochs by default; C1 and P1 use 50. All other model defaults are shared. Checkpoints use each dataset's own number and order of genes.

```bash
python -m geneflowv2.datasets download --dataset C2 --data-root data
python -m geneflowv2.run train --dataset C2 --data-root data
python -m geneflowv2.run validate --dataset C2 --data-root data

python -m geneflowv2.datasets download --dataset P1 --data-root data
python -m geneflowv2.run train --dataset P1 --data-root data
python -m geneflowv2.run validate --dataset P1 --data-root data
```

Alternatively, use `bash scripts/quick_start.sh C2` or `bash scripts/quick_start.sh P1`. Large P1 runs may require a smaller batch size; the ranking loss requires at least two cells per batch.

## Already downloaded data

For an existing public archive, verify and extract it with:

```bash
python -m geneflowv2.datasets download --dataset C1 \
  --archive /path/to/Xenium_V1_hSkin_Melanoma_Base_FFPE.tar.gz --data-root data
```

For an already extracted sample, prepare a portable training view without downloading again:

```bash
python -m geneflowv2.datasets prepare --dataset C1 \
  --sample-dir /path/to/Xenium_V1_hSkin_Melanoma_Base_FFPE --data-root data
python -m geneflowv2.datasets prepare --dataset C2 \
  --sample-dir /path/to/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE --data-root data
python -m geneflowv2.datasets prepare --dataset P1 \
  --sample-dir /path/to/Xenium_Prime_Human_Skin_FFPE --data-root data
```

`--sample-dir` must directly contain `adata.h5ad` and `cell_patch_256_aux/input/`. If you used `tar` manually, the archive may add a leading `processed_data/` directory; include that level in `--sample-dir`. The helper accepts both archive layouts and normalizes them into `data/raw/<sample>/`.

Preparation matches released image basenames to local files, falls back to `<cell_id>_original.tif`, retains cells with expression, images, and finite spatial coordinates, and records excluded-cell counts in `dataset.json`. It rejects duplicate cell/gene names and a split too small for ranking. The manifest records artifact hashes, gene order, retained counts, and split provenance. To decode every retained image during a check, use `--image-checks -1`; by default only eight are decoded and all paths are checked.

## Small check using real C1 data

After C1 is extracted, run the actual loader, trainer, and validator on 32 real cells and 16 genes:

```bash
python -m geneflowv2.datasets prepare --dataset C1 \
  --sample-dir data/raw/Xenium_V1_hSkin_Melanoma_Base_FFPE \
  --output-dir data/prepared/C1-smoke --max-cells 32 --max-genes 16
python -m geneflowv2.run train --dataset C1 \
  --prepared-dir data/prepared/C1-smoke --output-dir outputs/C1-smoke --smoke
python -m geneflowv2.run validate --dataset C1 \
  --prepared-dir data/prepared/C1-smoke --output-dir outputs/C1-smoke --smoke
```

This creates a small expression file while referencing the real TIFF images. `--smoke` forces CPU execution, a reduced 16 × 16 model, one epoch, and two generation steps. The original 256-pixel crop size remains in the spatial-split calculation. These settings test execution and are not research-quality training. C2 and P1 support the same commands using their own sample directory and dataset name.

A separate synthetic check requires no dataset download:

```bash
bash scripts/smoke_test.sh
```

## Data conventions

The released C1/C2 TIFFs checked locally have six channels, and P1 has nine. The model reads the **first four channels** for all three datasets; the full files are preserved. Inputs are channel-last `[H, W, C]`; uint8 values are scaled to `[0, 1]`, while other dtypes receive per-image min-max conversion before scaling. Validation previews show only the first three channels.

Expression values from `adata.X` are used as released, with no implicit normalization or log transform. A measured zero and an unmeasured gene have different masks. The launcher preserves each panel's full gene order. Spatial coordinates come from `adata.obsm["spatial"]`, with an original crop width of 256 in the same pixel units. Each sample is split independently. The guard band removes training crops overlapping validation crops, so retained counts and fractions can differ from the input counts.

For custom AnnData/image inputs, the lower-level `python -m geneflowv2.train` and `python -m geneflowv2.validate` commands accept `--adata`, `--image_paths`, and `--split_manifest`. Their `--help` lists optional layer, panel, cell-type, and count filters. Apply the same preprocessing options to training and validation. Avoid `--nsamples_test` with a prepared full-data split; use a separately prepared subset instead.

## Architecture


Gene identity and expression values form tokens processed by a Transformer. All observed genes, including measured zeros, contribute to a pooled cell embedding. The embedding conditions a UNet through block-specific FiLM adapters. A cell-summary token and expression-selected gene tokens supply multi-scale spatial cross-attention. Genes absent from the assay are masked separately from measured zeros.

Training combines a sinusoidal rectified-flow objective with a low-time correct-versus-shuffled RNA ranking objective. Flow gradients update the full model; additional ranking gradients update only the RNA encoder, FiLM adapters, and spatial cross-attention path. Training includes condition dropout for classifier-free guidance, EMA weights, checkpoint resume, and a spatially separated validation split with a crop-overlap guard band.

Default routing is `hard` (top expressed genes). `--routing_mode soft` enables continuous expression weighting; `all_observed` is also supported. These modes keep the same parameter layout but change conditioning behavior.

## Training and checkpoint details

The default model has 128 UNet base channels, channel multipliers `[1, 2, 2, 2, 2]`, and a 3-layer Transformer with 256-dimensional gene tokens. Attention resolution arguments refer to downsampling factors. Best-checkpoint selection minimizes validation `main_loss + 5 * ranking_loss` using EMA weights. Additional conditioning gradients use a ranking weight of 0.5 by default. A final singleton batch is dropped because shuffling needs at least two cells.

To resume a C1 run:

```bash
python -m geneflowv2.run train --dataset C1 --data-root data -- --auto_resume
```

Repeat any overrides from the original run. The model, loss settings, gene order, and spatial split must match. `--epochs` is the total target count; early-stopped runs also need a larger `--patience` to continue. Keep the batch size, workers, seeds, and software environment fixed for reproducible continuation. Moving outputs to another machine is supported by the launcher, which supplies the currently prepared split path explicitly.

## Validation metrics

Reported quantities include correct, shuffled, and null-condition velocity MSE; low-time shuffled-minus-correct MSE; the fraction where correct RNA beats shuffled RNA; generated-image changes under RNA shuffling and a changed noise seed; routed gene counts; and gene attention. Positive shuffled-minus-correct MSE indicates an advantage for correct RNA. These are conditioning diagnostics, not FID, LPIPS, biological correctness, or causal validation.

Generation uses fixed-step Euler integration with classifier-free guidance. The preview columns are: real image, correct RNA, shuffled RNA, null condition, alternate noise, and normalized correct-versus-shuffled difference. Previews show the first three channels. Attention values are diagnostics and are not causal gene attributions.

## Source and checks

```text
geneflowv2/
├── model.py           # Model and expression routing
├── blocks.py          # Transformer, UNet, FiLM, attention
├── datasets.py        # C1/C2/P1 download, extraction, preparation, checks
├── run.py             # Dataset-aware train/validate launcher
├── train.py           # Losses, EMA, checkpoints, epoch validation
├── validate.py        # Held-out conditioning evaluation
├── data.py            # AnnData/TIFF/HDF5 loading
├── flow.py            # Flow path and shared helpers
└── spatial_split.py   # Spatial split and overlap guard
scripts/
├── quick_start.sh     # Full C1/C2/P1 experiment
├── smoke_test.sh      # Synthetic CPU check
└── make_demo_data.py
tests/
├── test_datasets.py
└── test_model.py
```

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
python -m geneflowv2.datasets --help
python -m geneflowv2.run --help
```

Local release checks cover all three real datasets' preparation and a reduced training/checkpoint/validation run on each. Unit tests cover path repair, differing TIFF channel counts, independent gene panels, data-tree relocation, archive extraction, dataset mismatch detection, model behavior, and conditioning gradients. These checks do not represent full training convergence or a fresh download of the complete archives.

All public names use GeneFlowV2. Checkpoints must match both its method and architecture identifiers. The ZIP includes source and scripts; public datasets are downloaded separately, and trained weights and experiment outputs are excluded. The existing MIT license and original copyright notice are retained in [LICENSE](LICENSE). The linked dataset remains a separately distributed upstream resource.
