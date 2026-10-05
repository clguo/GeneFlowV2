from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm


from .flow import (
    derangement_indices,
    load_image_paths,
    mse_per_sample,
    sample_rectified_path,
    sample_t,
    set_seed,
)
from .data import CellImageGeneDataset
from .model import GENEFLOWV2_ARCHITECTURE, GeneFlowV2Model
from .spatial_split import default_manifest_for_model, dataset_view, load_split_manifest
from .data import parse_adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GeneFlowV2 conditioning.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adata", required=True)
    parser.add_argument("--image_paths", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--gene_symbols", default=None)
    parser.add_argument("--missing_gene_symbols", default=None)
    parser.add_argument("--cell_type", nargs="*", default=None)
    parser.add_argument("--exclude_cell_type", nargs="*", default=None)
    parser.add_argument("--cell_type_label", default="cell_type")
    parser.add_argument("--min_total_counts", type=int, default=0)
    parser.add_argument("--max_total_counts", type=float, default=np.inf)
    parser.add_argument("--min_total_pct", type=float, default=0.0)
    parser.add_argument("--max_total_pct", type=float, default=1.0)
    parser.add_argument("--nsamples_test", type=int, default=-1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--img_channels", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_noise_seed", type=int, default=123)
    parser.add_argument("--alt_noise_seed", type=int, default=456)
    parser.add_argument("--gen_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--time_sampling_power", type=float, default=1.0)
    parser.add_argument("--stochastic_path_noise", type=float, default=0.05)
    parser.add_argument("--low_t_max", type=float, default=0.25)
    parser.add_argument("--preview_samples", type=int, default=8)
    return parser.parse_args()


def load_model(checkpoint, fallback_img_size: int, fallback_img_channels: int):
    if checkpoint.get("method") != "GeneFlowV2":
        raise ValueError("V2 evaluator requires a GeneFlowV2 checkpoint")
    if checkpoint.get("architecture") != GENEFLOWV2_ARCHITECTURE:
        raise ValueError("V2 evaluator received an incompatible architecture")
    config = dict(checkpoint["model_config"])
    config["img_size"] = config.get("img_size", fallback_img_size)
    config["img_channels"] = config.get("img_channels", fallback_img_channels)
    model = GeneFlowV2Model(**config)
    state = checkpoint.get("ema_model_state_dict")
    if state is None:
        raise ValueError("V2 checkpoint is missing EMA weights")
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model, config


def align_expression(expr_df, checkpoint):
    expected = [str(gene) for gene in checkpoint.get("gene_names", [])]
    current = [str(gene) for gene in expr_df.columns]
    if not expected:
        raise ValueError("V2 checkpoint does not contain gene_names")
    if current == expected:
        return expr_df
    if set(current) != set(expected):
        missing = sorted(set(expected).difference(current))
        extra = sorted(set(current).difference(expected))
        raise ValueError(
            f"Evaluation genes differ; missing={missing[:10]}, extra={extra[:10]}"
        )
    return expr_df.loc[:, expected]


@torch.no_grad()
def generate(
    model,
    gene_expr,
    gene_mask,
    initial_noise,
    num_steps: int,
    guidance_scale: float,
    unconditional: bool = False,
):
    model.eval()
    x = initial_noise.clone()
    batch_size = gene_expr.shape[0]
    dt = 1.0 / num_steps
    times = torch.linspace(0.0, 1.0 - dt, num_steps, device=x.device)
    absent = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    for time in times:
        t = torch.full((batch_size,), float(time), device=x.device)
        if unconditional:
            velocity = model(
                x,
                t,
                gene_expr,
                gene_mask=gene_mask,
                condition_present=absent,
            )
        else:
            velocity = model.forward_with_cfg(
                x,
                t,
                gene_expr,
                gene_mask=gene_mask,
                guidance_scale=guidance_scale,
            )
        x = x + velocity * dt
    return x.clamp(0, 1)


def l1_per_sample(first, second):
    return torch.abs(first - second).flatten(1).mean(dim=1)


def save_preview(path: Path, real, correct, shuffled, null, alternate):
    rows = []
    for index in range(min(real.shape[0], correct.shape[0])):
        delta = torch.abs(correct[index, :3] - shuffled[index, :3])
        if float(delta.max()) > 0:
            delta = delta / delta.max()
        rows.extend(
            [
                real[index, :3],
                correct[index, :3],
                shuffled[index, :3],
                null[index, :3],
                alternate[index, :3],
                delta,
            ]
        )
    grid = make_grid(torch.stack(rows), nrow=6, padding=8)
    image = (
        grid.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
    from PIL import Image

    Image.fromarray(image).save(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("conditioning evaluation requires batch_size >= 2")
    if not 0.0 < args.low_t_max <= 1.0:
        raise ValueError("low_t_max must be in (0, 1]")
    if args.gen_steps < 1 or args.preview_samples < 1:
        raise ValueError("gen_steps and preview_samples must be positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model, model_config = load_model(
        checkpoint, args.img_size, args.img_channels
    )
    model.to(device).eval()

    expr_df, missing_gene_symbols = parse_adata(args)
    expr_df.index = expr_df.index.astype(str)
    expr_df = align_expression(expr_df, checkpoint)
    dataset = CellImageGeneDataset(
        expr_df,
        load_image_paths(args.image_paths),
        img_size=model.img_size,
        img_channels=model.img_channels,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((model.img_size, model.img_size), antialias=True),
            ]
        ),
        missing_gene_symbols=missing_gene_symbols,
    )

    if args.split_manifest:
        split_path = Path(args.split_manifest)
    elif checkpoint.get("split_manifest_path"):
        split_path = Path(checkpoint["split_manifest_path"])
    else:
        split_path = default_manifest_for_model(args.model_path)
    split_manifest = load_split_manifest(split_path, dataset.cell_ids)
    if checkpoint.get("split_manifest_sha256") != split_manifest["manifest_sha256"]:
        raise ValueError("Checkpoint and evaluation split checksums differ")
    dataset = dataset_view(dataset, split_manifest["val_cell_ids"])

    sample_count = (
        len(dataset) if args.num_samples < 0 else min(args.num_samples, len(dataset))
    )
    if sample_count < 2:
        raise ValueError("conditioning evaluation requires at least two samples")
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    )[:sample_count].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=sample_count % args.batch_size == 1,
    )

    rows = []
    attention_rows = []
    preview = None
    for batch_index, batch in enumerate(tqdm(loader, desc="GeneFlowV2 conditioning")):
        expression = batch["gene_expr"].to(device)
        real = batch["image"].to(device)
        gene_mask = batch.get("gene_mask")
        if gene_mask is not None:
            gene_mask = gene_mask.to(device)
        sample_ids = [str(cell_id) for cell_id in batch["cell_id"]]
        batch_size = expression.shape[0]
        present = torch.ones(batch_size, dtype=torch.bool, device=device)
        absent = torch.zeros(batch_size, dtype=torch.bool, device=device)
        permutation = derangement_indices(batch_size, device)
        shuffled_expression = expression[permutation]
        shuffled_mask = gene_mask[permutation] if gene_mask is not None else None

        t = sample_t(batch_size, device, args.time_sampling_power)
        x_t, target_velocity = sample_rectified_path(
            real,
            t,
            shared_noise=True,
            stochastic_path_noise=args.stochastic_path_noise,
        )
        low_t = torch.rand(batch_size, device=device) * args.low_t_max
        low_x_t, low_target_velocity = sample_rectified_path(
            real,
            low_t,
            shared_noise=True,
            stochastic_path_noise=args.stochastic_path_noise,
        )

        with torch.no_grad():
            (
                v_correct,
                cell_token_attention,
                spatial_gene_attention,
                active_gene_mask,
            ) = model(
                x_t,
                t,
                expression,
                gene_mask,
                present,
                return_cross_attention=True,
            )
            v_shuffled = model(x_t, t, shuffled_expression, shuffled_mask, present)
            v_null = model(x_t, t, expression, gene_mask, absent)
            correct_mse = mse_per_sample(v_correct, target_velocity)
            shuffled_mse = mse_per_sample(v_shuffled, target_velocity)
            null_mse = mse_per_sample(v_null, target_velocity)

            low_correct = model(
                low_x_t, low_t, expression, gene_mask, present
            )
            low_shuffled = model(
                low_x_t, low_t, shuffled_expression, shuffled_mask, present
            )
            low_correct_mse = mse_per_sample(low_correct, low_target_velocity)
            low_shuffled_mse = mse_per_sample(low_shuffled, low_target_velocity)

            fixed_generator = torch.Generator(device=device).manual_seed(
                args.fixed_noise_seed + batch_index
            )
            alternate_generator = torch.Generator(device=device).manual_seed(
                args.alt_noise_seed + batch_index
            )
            shape = (
                batch_size,
                model.img_channels,
                model.img_size,
                model.img_size,
            )
            fixed_noise = torch.randn(
                shape, generator=fixed_generator, device=device, dtype=expression.dtype
            )
            alternate_noise = torch.randn(
                shape,
                generator=alternate_generator,
                device=device,
                dtype=expression.dtype,
            )
            generated_correct = generate(
                model,
                expression,
                gene_mask,
                fixed_noise,
                args.gen_steps,
                args.guidance_scale,
            )
            generated_shuffled = generate(
                model,
                shuffled_expression,
                shuffled_mask,
                fixed_noise,
                args.gen_steps,
                args.guidance_scale,
            )
            generated_null = generate(
                model,
                expression,
                gene_mask,
                fixed_noise,
                args.gen_steps,
                1.0,
                unconditional=True,
            )
            generated_alternate = generate(
                model,
                expression,
                gene_mask,
                alternate_noise,
                args.gen_steps,
                args.guidance_scale,
            )
            conditioning = model.rna_encoder.encode_conditioning(
                expression, gene_mask=gene_mask, return_attention=True
            )

        shuffled_l1 = l1_per_sample(generated_correct, generated_shuffled)
        null_l1 = l1_per_sample(generated_correct, generated_null)
        seed_l1 = l1_per_sample(generated_correct, generated_alternate)
        for index, cell_id in enumerate(sample_ids):
            rows.append(
                {
                    "sample_id": cell_id,
                    "correct_velocity_mse": float(correct_mse[index].cpu()),
                    "shuffled_velocity_mse": float(shuffled_mse[index].cpu()),
                    "null_velocity_mse": float(null_mse[index].cpu()),
                    "shuffle_minus_correct_mse": float(
                        (shuffled_mse[index] - correct_mse[index]).cpu()
                    ),
                    "null_minus_correct_mse": float(
                        (null_mse[index] - correct_mse[index]).cpu()
                    ),
                    "low_t_correct_velocity_mse": float(low_correct_mse[index].cpu()),
                    "low_t_shuffled_velocity_mse": float(low_shuffled_mse[index].cpu()),
                    "low_t_shuffle_minus_correct_mse": float(
                        (low_shuffled_mse[index] - low_correct_mse[index]).cpu()
                    ),
                    "same_seed_true_vs_shuffled_l1": float(shuffled_l1[index].cpu()),
                    "same_seed_true_vs_null_l1": float(null_l1[index].cpu()),
                    "same_rna_seed_effect_l1": float(seed_l1[index].cpu()),
                    "rna_shuffle_to_seed_ratio": float(
                        (shuffled_l1[index] / (seed_l1[index] + 1e-8)).cpu()
                    ),
                    "active_gene_count": int(active_gene_mask[index].sum().cpu()),
                    "cell_token_cross_attention": float(
                        cell_token_attention[index].cpu()
                    ),
                }
            )

            pooling_weights = conditioning.pooling_attention[index].cpu().numpy()
            spatial_weights = spatial_gene_attention[index].cpu().numpy()
            active_values = active_gene_mask[index].cpu().numpy()
            expression_values = expression[index].cpu().numpy()
            for gene, expr, pooling, spatial, active in zip(
                checkpoint["gene_names"],
                expression_values,
                pooling_weights,
                spatial_weights,
                active_values,
            ):
                attention_rows.append(
                    {
                        "sample_id": cell_id,
                        "gene": str(gene),
                        "expression": float(expr),
                        "active_for_cross_attention": int(active),
                        "pooling_attention": float(pooling),
                        "spatial_cross_attention": float(spatial),
                    }
                )

        if preview is None:
            n = min(args.preview_samples, batch_size)
            preview = tuple(
                tensor[:n].detach().cpu()
                for tensor in (
                    real,
                    generated_correct,
                    generated_shuffled,
                    generated_null,
                    generated_alternate,
                )
            )

    metrics_path = output_dir / "geneflowv2_conditioning_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "geneflowv2_gene_attention.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attention_rows[0]))
        writer.writeheader()
        writer.writerows(attention_rows)

    summary = {
        "method": "GeneFlowV2",
        "architecture": GENEFLOWV2_ARCHITECTURE,
        "weights": "ema_model_state_dict",
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "model_path": args.model_path,
        "model_config": model_config,
        "training_config": checkpoint.get("training_config", {}),
        "n_samples": len(rows),
        "split": "val",
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "evaluation_config": {
            "seed": args.seed,
            "fixed_noise_seed": args.fixed_noise_seed,
            "alt_noise_seed": args.alt_noise_seed,
            "gen_steps": args.gen_steps,
            "guidance_scale": args.guidance_scale,
            "time_sampling_power": args.time_sampling_power,
            "stochastic_path_noise": args.stochastic_path_noise,
            "low_t_max": args.low_t_max,
        },
    }
    for key in rows[0]:
        if key == "sample_id":
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
        }
    summary["uniform_correct_better_fraction"] = float(
        np.mean([row["shuffle_minus_correct_mse"] > 0 for row in rows])
    )
    summary["low_t_correct_better_fraction"] = float(
        np.mean([row["low_t_shuffle_minus_correct_mse"] > 0 for row in rows])
    )
    with (output_dir / "geneflowv2_conditioning_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    if preview is not None:
        save_preview(output_dir / "geneflowv2_conditioning_preview.png", *preview)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
