from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Dict
import numpy as np
import torch
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_image_paths(path: str) -> Dict[str, str]:
    with open(path) as handle:
        image_paths = json.load(handle)
    base = Path(path).resolve().parent
    resolved = {}
    for key, value in image_paths.items():
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Image for cell {key} does not exist: {candidate}")
        resolved[str(key)] = str(candidate)
    return resolved

def sample_t(batch_size: int, device: torch.device, power: float) -> torch.Tensor:
    if power <= 0:
        raise ValueError(f"time_sampling_power must be positive, got {power}")
    u = torch.rand(batch_size, device=device)
    return u.pow(power).clamp(0.0, 1.0)

def sample_rectified_path(
    x_1: torch.Tensor,
    t: torch.Tensor,
    shared_noise: bool,
    stochastic_path_noise: float,
):

    if shared_noise:
        start_noise = torch.randn_like(x_1[:1]).expand_as(x_1).contiguous()
        stochastic = torch.randn_like(x_1[:1]).expand_as(x_1).contiguous()
    else:
        start_noise = torch.randn_like(x_1)
        stochastic = torch.randn_like(x_1)

    t_expanded = t.view(-1, *([1] * (x_1.ndim - 1)))
    interp_coef = torch.sin(t_expanded * (math.pi / 2))
    x_t = interp_coef * x_1 + (1.0 - interp_coef) * start_noise
    x_t = x_t + stochastic * ((1.0 - t_expanded) * stochastic_path_noise)
    velocity = (x_1 - start_noise) * (math.pi / 2) * torch.cos(
        t_expanded * (math.pi / 2)
    )

    velocity = velocity - stochastic * stochastic_path_noise
    return x_t, velocity

def mse_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).flatten(1).mean(dim=1)

def derangement_indices(batch_size: int, device: torch.device) -> torch.Tensor:

    indices = torch.arange(batch_size, device=device)
    if batch_size <= 1:
        return indices
    for _ in range(32):
        permutation = torch.randperm(batch_size, device=device)
        if not torch.any(permutation == indices):
            return permutation
    shift = int(torch.randint(1, batch_size, (), device=device).item())
    return (indices + shift) % batch_size

def update_ema(ema_model, model, decay: float) -> None:

    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
    model_state = model.state_dict()
    for name, ema_value in ema_model.state_dict().items():
        source = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            ema_value.copy_(source)
