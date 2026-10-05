from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from .flow import (
    derangement_indices,
    load_image_paths,
    mse_per_sample,
    sample_rectified_path,
    sample_t,
    set_seed,
    update_ema,
)
from .data import CellImageGeneDataset
from .model import (
    GENEFLOWV2_ARCHITECTURE,
    GENEFLOWV2_ROUTING_MODES,
    GeneFlowV2Model,
    geneflowv2_config_from_args,
)
from .spatial_split import (
    create_spatial_split_manifest,
    load_spatial_coordinates,
    load_split_manifest,
    save_split_manifest,
    split_dataset_views,
)
from .data import parse_adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train GeneFlowV2 with expression-routed gene-token attention and "
            "condition-only low-t ranking."
        )
    )
    parser.add_argument("--adata", required=True)
    parser.add_argument("--image_paths", required=True)
    parser.add_argument("--output_dir", required=True)


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument(
        "--amp_dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--spatial_key", default="spatial")
    parser.add_argument("--spatial_patch_size", type=float, default=None)

    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=[16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2, 2]
    )
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_head_channels", type=int, default=16)
    parser.add_argument("--rna_strength", type=float, default=1.0)
    parser.add_argument("--use_checkpoint", action="store_true", default=False)

    parser.add_argument("--gene_token_dim", type=int, default=256)
    parser.add_argument("--gene_transformer_layers", type=int, default=3)
    parser.add_argument("--gene_transformer_heads", type=int, default=8)
    parser.add_argument("--gene_transformer_ff_dim", type=int, default=1024)
    parser.add_argument("--gene_transformer_dropout", type=float, default=0.0)
    parser.add_argument(
        "--max_cross_attention_genes",
        type=int,
        default=128,
        help="Maximum active gene tokens routed to image attention; 0 means no cap.",
    )
    parser.add_argument(
        "--cross_attention_expression_threshold",
        type=float,
        default=0.0,
        help="Route observed genes only when abs(expression) exceeds this value.",
    )
    parser.add_argument(
        "--routing_mode",
        choices=GENEFLOWV2_ROUTING_MODES,
        default="hard",
        help=(
            "hard reproduces V2 top-k routing; soft uses a continuous "
            "expression-weighted attention prior without top-k; "
            "all_observed routes every assayed gene and is the exact "
            "no-expression-routing ablation."
        ),
    )
    parser.add_argument(
        "--soft_routing_temperature",
        type=float,
        default=1.0,
        help=(
            "Expression scale in 1-exp(-abs(expression)/temperature); "
            "used only when --routing_mode soft."
        ),
    )
    parser.add_argument(
        "--cross_attention_resolutions", type=int, nargs="+", default=[8, 16]
    )
    parser.add_argument("--cross_attention_heads", type=int, default=8)
    parser.add_argument("--cross_attention_head_dim", type=int, default=32)
    parser.add_argument("--cross_attention_dropout", type=float, default=0.0)
    parser.add_argument("--cross_attention_gate_init", type=float, default=0.1)

    parser.add_argument("--condition_dropout_prob", type=float, default=0.15)
    parser.add_argument(
        "--condition_ranking_weight",
        type=float,
        default=0.5,
        help="Weight of ranking gradients added only to conditioning parameters.",
    )
    parser.add_argument(
        "--ranking_margin",
        type=float,
        default=0.001,
        help="Desired shuffled-minus-correct velocity MSE on the low-t branch.",
    )
    parser.add_argument(
        "--ranking_temperature",
        type=float,
        default=0.001,
        help="Smooth-hinge temperature for the ranking objective.",
    )
    parser.add_argument(
        "--ranking_time_max",
        type=float,
        default=0.25,
        help="Ranking states sample t uniformly from [0, this value].",
    )
    parser.add_argument(
        "--selection_ranking_weight",
        type=float,
        default=5.0,
        help="Weight of val ranking loss in best-checkpoint selection only.",
    )
    parser.add_argument("--shared_noise_prob", type=float, default=0.75)
    parser.add_argument("--time_sampling_power", type=float, default=1.0)
    parser.add_argument("--stochastic_path_noise", type=float, default=0.05)

    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help=(
            "Overwrite latest_checkpoint.pt every N successful optimizer steps; "
            "0 disables mid-epoch checkpoints."
        ),
    )
    parser.add_argument("--skip_nonfinite_batches", action="store_true", default=True)
    parser.add_argument(
        "--no_skip_nonfinite_batches",
        dest="skip_nonfinite_batches",
        action="store_false",
    )
    parser.add_argument("--abort_on_nonfinite", action="store_true", default=False)
    return parser.parse_args()


def training_config_from_args(args: argparse.Namespace) -> dict:

    keys = {
        "condition_dropout_prob",
        "condition_ranking_weight",
        "ranking_margin",
        "ranking_temperature",
        "ranking_time_max",
        "selection_ranking_weight",
        "shared_noise_prob",
        "time_sampling_power",
        "stochastic_path_noise",
        "ema_decay",
    }
    return {key: value for key, value in vars(args).items() if key in keys}


def _batch_tensors(batch, device: torch.device):
    expression = batch["gene_expr"].to(device)
    images = batch["image"].to(device)
    gene_mask = batch.get("gene_mask")
    if gene_mask is not None:
        gene_mask = gene_mask.to(device)
    return expression, images, gene_mask


def _amp_context(args, device: torch.device):
    enabled = bool(args.use_amp and device.type == "cuda")
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)


def compute_main_flow_loss(
    model: GeneFlowV2Model,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
):
    expression, images, gene_mask = _batch_tensors(batch, device)
    batch_size = expression.shape[0]
    t = sample_t(batch_size, device, args.time_sampling_power)
    shared_noise = torch.rand((), device=device).item() < args.shared_noise_prob
    x_t, target_velocity = sample_rectified_path(
        images,
        t,
        shared_noise=shared_noise,
        stochastic_path_noise=args.stochastic_path_noise,
    )

    with _amp_context(args, device):
        if train and args.condition_dropout_prob > 0:
            condition_present = (
                torch.rand(batch_size, device=device) >= args.condition_dropout_prob
            )
            v_main = model(
                x_t,
                t,
                expression,
                gene_mask=gene_mask,
                condition_present=condition_present,
            )
            main_per_sample = mse_per_sample(v_main, target_velocity)
            main_loss = main_per_sample.mean()
            correct_rows = condition_present
            correct_loss = (
                main_per_sample[correct_rows].mean()
                if bool(correct_rows.any())
                else main_loss.detach()
            )
            null_rows = ~condition_present
            null_loss = (
                main_per_sample[null_rows].mean()
                if bool(null_rows.any())
                else torch.zeros((), device=device, dtype=main_loss.dtype)
            )
            null_fraction = null_rows.float().mean()
        else:
            all_present = torch.ones(batch_size, dtype=torch.bool, device=device)
            v_correct = model(
                x_t,
                t,
                expression,
                gene_mask=gene_mask,
                condition_present=all_present,
            )
            correct_per_sample = mse_per_sample(v_correct, target_velocity)
            correct_loss = correct_per_sample.mean()
            main_loss = correct_loss
            absent = torch.zeros(batch_size, dtype=torch.bool, device=device)
            v_null = model(
                x_t,
                t,
                expression,
                gene_mask=gene_mask,
                condition_present=absent,
            )
            null_loss = mse_per_sample(v_null, target_velocity).mean()
            null_fraction = torch.zeros((), device=device)

    with torch.no_grad():
        observed = (
            torch.ones_like(expression, dtype=torch.bool)
            if gene_mask is None
            else gene_mask > 0
        )
        gene_routing = model.rna_encoder.build_spatial_gene_routing(
            expression, observed
        )
        active_count = (gene_routing > 0).sum(dim=1).float().mean()
    metrics = {
        "main_loss": main_loss.detach(),
        "correct_loss": correct_loss.detach(),
        "null_loss": null_loss.detach(),
        "null_fraction": null_fraction.detach(),
        "active_gene_count": active_count.detach(),
        "cross_attention_gate": model.cross_attention_gate_values().mean().detach(),
    }
    return main_loss, metrics


def compute_condition_ranking_loss(
    model: GeneFlowV2Model,
    batch,
    args: argparse.Namespace,
    device: torch.device,
):
    expression, images, gene_mask = _batch_tensors(batch, device)
    batch_size = expression.shape[0]
    if batch_size <= 1:
        zero = images.sum() * 0.0
        return zero, {
            "ranking_loss": zero.detach(),
            "ranking_correct_loss": zero.detach(),
            "ranking_shuffle_loss": zero.detach(),
            "ranking_gap": zero.detach(),
            "ranking_accuracy": zero.detach(),
        }

    permutation = derangement_indices(batch_size, device)
    shuffled_expression = expression[permutation]
    shuffled_mask = gene_mask[permutation] if gene_mask is not None else None

    t = torch.rand(batch_size, device=device) * args.ranking_time_max
    x_t, target_velocity = sample_rectified_path(
        images,
        t,
        shared_noise=True,
        stochastic_path_noise=args.stochastic_path_noise,
    )
    present = torch.ones(batch_size, dtype=torch.bool, device=device)

    with _amp_context(args, device):
        v_correct = model(
            x_t,
            t,
            expression,
            gene_mask=gene_mask,
            condition_present=present,
        )
        v_shuffled = model(
            x_t,
            t,
            shuffled_expression,
            gene_mask=shuffled_mask,
            condition_present=present,
        )
        correct_per_sample = mse_per_sample(v_correct, target_velocity)
        shuffled_per_sample = mse_per_sample(v_shuffled, target_velocity)
        gap = shuffled_per_sample - correct_per_sample
        ranking_loss = (
            F.softplus(
                (args.ranking_margin - gap) / args.ranking_temperature
            )
            * args.ranking_temperature
        ).mean()

    metrics = {
        "ranking_loss": ranking_loss.detach(),
        "ranking_correct_loss": correct_per_sample.mean().detach(),
        "ranking_shuffle_loss": shuffled_per_sample.mean().detach(),
        "ranking_gap": gap.mean().detach(),
        "ranking_accuracy": (gap > 0).float().mean().detach(),
    }
    return ranking_loss, metrics


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def add_condition_only_ranking_gradients(
    model: GeneFlowV2Model,
    weighted_ranking_loss: torch.Tensor,
) -> int:

    parameters = list(model.conditioning_parameters())
    gradients = torch.autograd.grad(
        weighted_ranking_loss,
        parameters,
        allow_unused=True,
    )
    applied = 0
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient
        applied += 1
    if applied == 0:
        raise RuntimeError("V2 ranking loss produced no conditioning gradients")
    return applied


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    args,
    device,
    train: bool,
    ema_model=None,
    start_batch_index: int = 0,
    initial_state: Optional[dict] = None,
    optimizer_step_callback: Optional[Callable[[int, dict], None]] = None,
):
    model.train(train)
    initial_state = initial_state or {}
    totals: Dict[str, float] = {
        str(name): float(value)
        for name, value in initial_state.get("totals", {}).items()
    }
    n_batches = int(initial_state.get("n_batches", 0))
    skipped_nonfinite = int(initial_state.get("skipped_nonfinite", 0))
    iterator = tqdm(loader, desc="train" if train else "val")

    for batch_index, batch in enumerate(iterator):
        if batch_index < start_batch_index:
            continue
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            main_loss, main_metrics = compute_main_flow_loss(
                model, batch, args, device, train=train
            )
            if not _all_finite([main_loss, *main_metrics.values()]):
                skipped_nonfinite += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                message = f"Non-finite V2 main loss at batch {batch_index}"
                if args.abort_on_nonfinite or not args.skip_nonfinite_batches:
                    raise FloatingPointError(message)
                continue

            if train:
                if scaler is not None:
                    scaler.scale(main_loss).backward()
                else:
                    main_loss.backward()


            with torch.set_grad_enabled(
                train and args.condition_ranking_weight > 0
            ):
                ranking_loss, ranking_metrics = compute_condition_ranking_loss(
                    model, batch, args, device
                )
            if not _all_finite([ranking_loss, *ranking_metrics.values()]):
                skipped_nonfinite += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                message = f"Non-finite V2 ranking loss at batch {batch_index}"
                if args.abort_on_nonfinite or not args.skip_nonfinite_batches:
                    raise FloatingPointError(message)
                continue

            if train and args.condition_ranking_weight > 0:
                weighted = args.condition_ranking_weight * ranking_loss
                if scaler is not None:
                    weighted = scaler.scale(weighted)
                add_condition_only_ranking_gradients(model, weighted)

            if train:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.update()
                    raise FloatingPointError(
                        f"Non-finite GeneFlowV2 gradient at batch {batch_index}"
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if ema_model is not None:
                    update_ema(ema_model, model, args.ema_decay)

        loss = main_loss.detach() + args.condition_ranking_weight * ranking_loss.detach()
        selection_score = (
            main_loss.detach()
            + args.selection_ranking_weight * ranking_loss.detach()
        )
        metrics = {
            "loss": loss,
            "selection_score": selection_score,
            **main_metrics,
            **ranking_metrics,
        }
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.cpu())
        n_batches += 1

        if train and optimizer_step_callback is not None:
            optimizer_step_callback(
                batch_index + 1,
                {
                    "totals": dict(totals),
                    "n_batches": n_batches,
                    "skipped_nonfinite": skipped_nonfinite,
                },
            )

        if train and (batch_index + 1) % args.log_every == 0:
            iterator.set_postfix(
                flow=f"{totals['main_loss'] / n_batches:.4f}",
                gap=f"{totals['ranking_gap'] / n_batches:.3e}",
                acc=f"{totals['ranking_accuracy'] / n_batches:.2f}",
                active=f"{totals['active_gene_count'] / n_batches:.0f}",
            )

    if n_batches == 0:
        raise FloatingPointError(
            f"All {'train' if train else 'validation'} batches were non-finite"
        )
    results = {name: value / n_batches for name, value in totals.items()}
    results["skipped_nonfinite"] = skipped_nonfinite
    return results


def save_checkpoint(
    path: Path,
    model,
    ema_model,
    optimizer,
    scheduler,
    epoch: int,
    metrics,
    model_config,
    training_config,
    gene_names,
    split_metadata,
    best_selection_score: float,
    best_flow_loss: float,
    patience_counter: int,
    global_step: int = 0,
    next_batch_index: int = 0,
    epoch_complete: bool = True,
    train_epoch_state: Optional[dict] = None,
) -> None:
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "next_batch_index": next_batch_index,
            "global_step": global_step,
            "train_epoch_state": train_epoch_state,
            "rng_state": rng_state,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "best_selection_score": best_selection_score,
            "best_flow_loss": best_flow_loss,
            "patience_counter": patience_counter,
            "model_config": model_config,
            "training_config": training_config,
            "gene_names": gene_names,
            "method": "GeneFlowV2",
            "architecture": GENEFLOWV2_ARCHITECTURE,
            "split_manifest_path": split_metadata["path"],
            "split_manifest_sha256": split_metadata["sha256"],
            "split_strategy": split_metadata["strategy"],
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def restore_rng_state(rng_state: Optional[dict]) -> None:

    if not rng_state:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])


    torch_rng_state = torch.as_tensor(
        rng_state["torch"], dtype=torch.uint8, device="cpu"
    ).contiguous()
    torch.set_rng_state(torch_rng_state)
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        cuda_rng_states = [
            torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()
            for state in rng_state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 and args.condition_ranking_weight > 0:
        raise ValueError("V2 condition ranking requires batch_size >= 2")
    if not 0.0 < args.ranking_time_max <= 1.0:
        raise ValueError("ranking_time_max must be in (0, 1]")
    if args.ranking_temperature <= 0:
        raise ValueError("ranking_temperature must be positive")
    if args.soft_routing_temperature <= 0:
        raise ValueError("soft_routing_temperature must be positive")
    if args.condition_ranking_weight < 0 or args.selection_ranking_weight < 0:
        raise ValueError("ranking weights must be non-negative")
    if args.save_every_steps < 0 or args.save_every_epochs < 0:
        raise ValueError("checkpoint intervals must be non-negative")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    expr_df, missing_gene_symbols = parse_adata(args)
    expr_df.index = expr_df.index.astype(str)
    gene_names = [str(gene) for gene in expr_df.columns]
    dataset = CellImageGeneDataset(
        expr_df,
        load_image_paths(args.image_paths),
        img_size=args.img_size,
        img_channels=args.img_channels,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size), antialias=True),
            ]
        ),
        missing_gene_symbols=missing_gene_symbols,
    )
    if len(dataset) == 0:
        raise RuntimeError("No cells have both RNA and an existing image path")

    split_path = (
        Path(args.split_manifest)
        if args.split_manifest
        else output_dir / "split_manifest.json"
    )
    if split_path.exists():
        split_manifest = load_split_manifest(split_path, dataset.cell_ids)
        print(f"Reusing spatial split manifest: {split_path}")
    else:
        coordinates, missing_coordinate_ids = load_spatial_coordinates(
            args.adata, dataset.cell_ids, spatial_key=args.spatial_key
        )
        split_manifest = create_spatial_split_manifest(
            dataset.cell_ids,
            coordinates,
            val_fraction=args.val_fraction,
            patch_size_pixels=args.spatial_patch_size or args.img_size,
            seed=args.seed,
            spatial_key=args.spatial_key,
        )
        save_split_manifest(split_manifest, split_path)
        print(f"Created spatial split manifest: {split_path}")
        if missing_coordinate_ids:
            print(f"Dropped {len(missing_coordinate_ids)} cells without coordinates")

    train_dataset, val_dataset = split_dataset_views(dataset, split_manifest)
    split_metadata = {
        "path": str(split_path.resolve()),
        "sha256": split_manifest["manifest_sha256"],
        "strategy": split_manifest["strategy"],
    }
    print(
        f"Spatial split train={len(train_dataset)}, val={len(val_dataset)}, "
        f"buffer_dropped={split_manifest['buffer_dropped_count']}, "
        f"overlap_audit={split_manifest['overlap_audit_train_cells_with_val_overlap']}"
    )

    train_loader_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) % args.batch_size == 1,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(val_dataset) % args.batch_size == 1,
    )

    model_config = geneflowv2_config_from_args(args, rna_dim=expr_df.shape[1])
    training_config = training_config_from_args(args)


    model = GeneFlowV2Model(**model_config).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    condition_count = sum(
        parameter.numel() for parameter in model.conditioning_parameters()
    )
    print(
        f"V2 architecture={GENEFLOWV2_ARCHITECTURE}, "
        f"parameters={parameter_count / 1e6:.2f}M, "
        f"conditioning_parameters={condition_count / 1e6:.2f}M, "
        f"cross_attention_blocks={model.unet.cross_attention_block_count}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if args.use_amp and args.amp_dtype == "float16" and device.type == "cuda"
        else None
    )

    start_epoch = 0
    best_selection = float("inf")
    best_flow = float("inf")
    patience_counter = 0
    global_step = 0
    resume_batch_index = 0
    resume_train_state = None
    resume_rng_state = None
    resume_checkpoint = args.resume_from
    if resume_checkpoint is None and args.auto_resume:
        candidate = checkpoint_dir / "latest_checkpoint.pt"
        if candidate.exists():
            resume_checkpoint = str(candidate)

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("method") != "GeneFlowV2":
            raise ValueError("GeneFlowV2 refuses to resume a non-V2 checkpoint")
        if checkpoint.get("architecture") != GENEFLOWV2_ARCHITECTURE:
            raise ValueError("GeneFlowV2 checkpoint architecture is incompatible")
        checkpoint_model_config = dict(checkpoint.get("model_config", {}))


        checkpoint_model_config.setdefault("routing_mode", "hard")
        checkpoint_model_config.setdefault("soft_routing_temperature", 1.0)
        if checkpoint_model_config != model_config:
            raise ValueError("V2 checkpoint model_config differs from this run")
        checkpoint_training_config = checkpoint.get("training_config")
        if checkpoint_training_config is None:
            raise ValueError("V2 checkpoint is missing training_config")
        if checkpoint_training_config != training_config:
            raise ValueError("V2 checkpoint training_config differs from this run")
        if checkpoint.get("split_manifest_sha256") != split_metadata["sha256"]:
            raise ValueError("V2 checkpoint was trained with a different spatial split")
        if [str(gene) for gene in checkpoint.get("gene_names", [])] != gene_names:
            raise ValueError("V2 checkpoint gene order differs from the AnnData")

        model_state = {
            key.replace("module.", ""): value
            for key, value in checkpoint["model_state_dict"].items()
        }
        ema_state = checkpoint.get("ema_model_state_dict")
        if ema_state is None:
            raise ValueError("V2 checkpoint is missing EMA weights")
        ema_state = {
            key.replace("module.", ""): value for key, value in ema_state.items()
        }
        model.load_state_dict(model_state, strict=True)
        ema_model.load_state_dict(ema_state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint["scheduler_state_dict"]
        saved_scheduler_epochs = int(scheduler_state.get("T_max", args.epochs))
        scheduler.load_state_dict(scheduler_state)
        if args.epochs > saved_scheduler_epochs:


            scheduler.T_max = args.epochs
            print(
                "Extended cosine scheduler horizon from "
                f"{saved_scheduler_epochs} to {args.epochs} epochs; "
                "continuing from the checkpoint learning rate"
            )
        checkpoint_epoch = int(checkpoint["epoch"])
        if bool(checkpoint.get("epoch_complete", True)):
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = checkpoint_epoch
            resume_batch_index = int(checkpoint.get("next_batch_index", 0))
            resume_train_state = checkpoint.get("train_epoch_state")
        global_step = int(
            checkpoint.get(
                "global_step",
                start_epoch * len(train_loader) + resume_batch_index,
            )
        )
        resume_rng_state = checkpoint.get("rng_state")
        best_selection = float(checkpoint.get("best_selection_score", float("inf")))
        best_flow = float(checkpoint.get("best_flow_loss", float("inf")))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        resume_location = f"epoch {start_epoch + 1}"
        if resume_batch_index:
            resume_location += f", batch {resume_batch_index + 1}"
        print(
            f"Resumed GeneFlowV2 from {resume_checkpoint} at {resume_location} "
            f"(global_step={global_step})"
        )

        if start_epoch >= args.epochs:
            print("Training is already complete; no additional epochs will run")
            return
        if patience_counter >= args.patience:
            print("Training already reached early stopping; no additional epochs will run")
            return

    with (output_dir / "geneflowv2_config.json").open("w") as handle:
        json.dump(
            {
                "architecture": GENEFLOWV2_ARCHITECTURE,
                "args": vars(args),
                "model_config": model_config,
                "training_config": training_config,
                "split_manifest": split_metadata,
            },
            handle,
            indent=2,
        )
    with (output_dir / "gene_names.json").open("w") as handle:
        json.dump(gene_names, handle, indent=2)

    restore_rng_state(resume_rng_state)

    history_path = output_dir / "training_history.csv"
    fieldnames = [
        "epoch",
        "split",
        "loss",
        "selection_score",
        "main_loss",
        "correct_loss",
        "null_loss",
        "ranking_loss",
        "ranking_correct_loss",
        "ranking_shuffle_loss",
        "ranking_gap",
        "ranking_accuracy",
        "null_fraction",
        "active_gene_count",
        "cross_attention_gate",
        "skipped_nonfinite",
        "lr",
    ]
    append_history = start_epoch > 0 and history_path.exists()
    with history_path.open("a" if append_history else "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not append_history:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            print(f"\nGeneFlowV2 epoch {epoch + 1}/{args.epochs}")
            train_loader_generator.manual_seed(args.seed + epoch)

            def checkpoint_after_optimizer_step(
                next_batch_index: int, train_epoch_state: dict
            ) -> None:
                nonlocal global_step
                global_step += 1
                if (
                    args.save_every_steps <= 0
                    or global_step % args.save_every_steps != 0
                ):
                    return
                save_checkpoint(
                    checkpoint_dir / "latest_checkpoint.pt",
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"train_partial": True},
                    model_config,
                    training_config,
                    gene_names,
                    split_metadata,
                    best_selection,
                    best_flow,
                    patience_counter,
                    global_step=global_step,
                    next_batch_index=next_batch_index,
                    epoch_complete=False,
                    train_epoch_state=train_epoch_state,
                )
                print(
                    f"Saved mid-epoch latest checkpoint at global step "
                    f"{global_step} (epoch {epoch + 1}, "
                    f"next batch {next_batch_index + 1})"
                )

            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                args,
                device,
                train=True,
                ema_model=ema_model,
                start_batch_index=resume_batch_index,
                initial_state=resume_train_state,
                optimizer_step_callback=checkpoint_after_optimizer_step,
            )
            resume_batch_index = 0
            resume_train_state = None
            val_metrics = run_epoch(
                ema_model, val_loader, None, None, args, device, train=False
            )
            scheduler.step()

            for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
                row = {
                    "epoch": epoch + 1,
                    "split": split_name,
                    "lr": optimizer.param_groups[0]["lr"],
                    **metrics,
                }
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            csv_file.flush()

            print(
                "train flow={:.5f} gap={:.3e} acc={:.3f}; "
                "val flow={:.5f} gap={:.3e} acc={:.3f} select={:.5f}".format(
                    train_metrics["main_loss"],
                    train_metrics["ranking_gap"],
                    train_metrics["ranking_accuracy"],
                    val_metrics["main_loss"],
                    val_metrics["ranking_gap"],
                    val_metrics["ranking_accuracy"],
                    val_metrics["selection_score"],
                )
            )

            best_flow = min(best_flow, val_metrics["main_loss"])
            if val_metrics["selection_score"] < best_selection:
                best_selection = val_metrics["selection_score"]
                patience_counter = 0
                save_checkpoint(
                    checkpoint_dir / "best_checkpoint.pt",
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    model_config,
                    training_config,
                    gene_names,
                    split_metadata,
                    best_selection,
                    best_flow,
                    patience_counter,
                    global_step=global_step,
                )
            else:
                patience_counter += 1

            save_checkpoint(
                checkpoint_dir / "latest_checkpoint.pt",
                model,
                ema_model,
                optimizer,
                scheduler,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                model_config,
                training_config,
                gene_names,
                split_metadata,
                best_selection,
                best_flow,
                patience_counter,
                global_step=global_step,
            )
            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    model_config,
                    training_config,
                    gene_names,
                    split_metadata,
                    best_selection,
                    best_flow,
                    patience_counter,
                    global_step=global_step,
                )

            if patience_counter >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement")
                break


if __name__ == "__main__":
    main()
