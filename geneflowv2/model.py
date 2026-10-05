from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import (
    GeneTokenTransformerEncoder,
    RNACrossAttentionUNet,
    SpatialGeneCrossAttention,
)


GENEFLOWV2_ARCHITECTURE = (
    "gene_token_transformer_expression_routed_cross_attention_"
    "condition_only_ranking"
)
GENEFLOWV2_ROUTING_MODES = ("hard", "soft", "all_observed")


@dataclass
class RoutedRNAConditioningV2:


    global_embedding: torch.Tensor
    spatial_tokens: torch.Tensor
    spatial_token_mask: torch.Tensor
    active_gene_mask: torch.Tensor
    pooling_attention: Optional[torch.Tensor] = None


class ExpressionRoutedGeneTokenEncoderV2(GeneTokenTransformerEncoder):


    def __init__(
        self,
        *args,
        max_cross_attention_genes: int = 128,
        cross_attention_expression_threshold: float = 0.0,
        routing_mode: str = "hard",
        soft_routing_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if max_cross_attention_genes < 0:
            raise ValueError("max_cross_attention_genes must be non-negative")
        if cross_attention_expression_threshold < 0:
            raise ValueError(
                "cross_attention_expression_threshold must be non-negative"
            )
        self.max_cross_attention_genes = int(max_cross_attention_genes)
        self.cross_attention_expression_threshold = float(
            cross_attention_expression_threshold
        )
        self.set_routing_mode(routing_mode, soft_routing_temperature)
        self.cell_token_projection = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.token_dim),
            nn.SiLU(),
            nn.LayerNorm(self.token_dim),
        )
        self.null_spatial_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
        nn.init.normal_(self.null_spatial_token, std=0.02)

    def set_routing_mode(
        self,
        routing_mode: str,
        soft_routing_temperature: float | None = None,
    ) -> None:


        if routing_mode not in GENEFLOWV2_ROUTING_MODES:
            choices = ", ".join(repr(mode) for mode in GENEFLOWV2_ROUTING_MODES)
            raise ValueError(f"routing_mode must be one of: {choices}")
        if soft_routing_temperature is not None:
            if soft_routing_temperature <= 0:
                raise ValueError("soft_routing_temperature must be positive")
            self.soft_routing_temperature = float(soft_routing_temperature)
        elif not hasattr(self, "soft_routing_temperature"):
            self.soft_routing_temperature = 1.0
        self.routing_mode = routing_mode

    def build_active_gene_mask(
        self,
        expression: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:

        clean_magnitude = torch.nan_to_num(expression).abs()
        eligible = observed & (
            clean_magnitude > self.cross_attention_expression_threshold
        )
        if self.max_cross_attention_genes == 0:
            return eligible

        top_k = min(self.max_cross_attention_genes, self.num_genes)
        if top_k == self.num_genes:
            return eligible
        scores = clean_magnitude.masked_fill(~eligible, float("-inf"))
        top_indices = scores.topk(top_k, dim=1, largest=True, sorted=False).indices
        selected = torch.zeros_like(eligible)
        selected.scatter_(1, top_indices, True)
        return eligible & selected

    def build_spatial_token_weights(
        self,
        expression: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:


        magnitude = torch.nan_to_num(expression).abs()
        weights = -torch.expm1(-magnitude / self.soft_routing_temperature)
        return weights.clamp(0.0, 1.0) * observed.to(weights.dtype)

    def build_spatial_gene_routing(
        self,
        expression: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:

        if self.routing_mode == "hard":
            return self.build_active_gene_mask(expression, observed)
        if self.routing_mode == "soft":
            return self.build_spatial_token_weights(expression, observed)


        return observed.clone()

    def encode_conditioning(
        self,
        expression: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
        condition_present: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> RoutedRNAConditioningV2:
        observed, present = self._validate_inputs(
            expression, gene_mask, condition_present
        )
        batch_size = expression.shape[0]


        clean_expression = torch.nan_to_num(expression) * observed.to(expression.dtype)
        value_tokens = self.expression_encoder(clean_expression.unsqueeze(-1))
        identity_tokens = self.gene_identity.unsqueeze(0).expand(batch_size, -1, -1)
        observation_tokens = self.observation_embedding(observed.long())
        tokens = self.input_dropout(
            self.input_norm(identity_tokens + value_tokens + observation_tokens)
        )

        has_observed_gene = observed.any(dim=1)
        safe_observed = observed.clone()
        safe_observed[~has_observed_gene, 0] = True
        key_padding_mask = ~safe_observed
        encoded_tokens = self.transformer(
            tokens, src_key_padding_mask=key_padding_mask
        )

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, attention = self.pool_attention(
            query,
            encoded_tokens,
            encoded_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        conditional_embedding = self.output_projection(pooled[:, 0])
        effective_present = present & has_observed_gene
        global_embedding = torch.where(
            effective_present[:, None],
            conditional_embedding,
            self.null_embedding.expand(batch_size, -1),
        )

        gene_routing = self.build_spatial_gene_routing(expression, observed)
        if gene_routing.dtype == torch.bool:
            gene_routing = gene_routing & effective_present[:, None]
            active_gene_mask = gene_routing
        else:
            gene_routing = gene_routing * effective_present[:, None].to(
                gene_routing.dtype
            )
            active_gene_mask = gene_routing > 0


        cell_token = self.cell_token_projection(conditional_embedding).unsqueeze(1)
        conditional_tokens = torch.cat([cell_token, encoded_tokens], dim=1)
        null_tokens = self.null_spatial_token.expand(
            batch_size, self.num_genes + 1, -1
        )
        spatial_tokens = torch.where(
            effective_present[:, None, None], conditional_tokens, null_tokens
        )
        cell_routing = torch.ones(
            batch_size, 1, dtype=gene_routing.dtype, device=expression.device
        )
        spatial_token_mask = torch.cat([cell_routing, gene_routing], dim=1)

        pooling_attention = None
        if return_attention:
            pooling_attention = attention[:, :, 0].mean(dim=1)
            pooling_attention = pooling_attention.masked_fill(~observed, 0.0)
            pooling_attention = torch.where(
                effective_present[:, None],
                pooling_attention,
                torch.zeros_like(pooling_attention),
            )

        return RoutedRNAConditioningV2(
            global_embedding=global_embedding,
            spatial_tokens=spatial_tokens,
            spatial_token_mask=spatial_token_mask,
            active_gene_mask=active_gene_mask,
            pooling_attention=pooling_attention,
        )

    def forward(
        self,
        expression: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
        condition_present: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        conditioning = self.encode_conditioning(
            expression,
            gene_mask=gene_mask,
            condition_present=condition_present,
            return_attention=return_attention,
        )
        if not return_attention:
            return conditioning.global_embedding
        return (
            conditioning.global_embedding,
            conditioning.spatial_tokens,
            conditioning.pooling_attention,
            conditioning.active_gene_mask,
        )


class GeneFlowV2Model(nn.Module):


    def __init__(
        self,
        rna_dim: int,
        img_channels: int = 4,
        img_size: int = 256,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions=(16,),
        dropout: float = 0.0,
        channel_mult=(1, 2, 2, 2, 2),
        use_checkpoint: bool = False,
        num_heads: int = 2,
        num_head_channels: int = 16,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_new_attention_order: bool = True,
        rna_strength: float = 1.0,
        gene_token_dim: int = 256,
        gene_transformer_layers: int = 3,
        gene_transformer_heads: int = 8,
        gene_transformer_ff_dim: int = 1024,
        gene_transformer_dropout: float = 0.0,
        max_cross_attention_genes: int = 128,
        cross_attention_expression_threshold: float = 0.0,
        routing_mode: str = "hard",
        soft_routing_temperature: float = 1.0,
        cross_attention_resolutions=(8, 16),
        cross_attention_heads: int = 8,
        cross_attention_head_dim: int = 32,
        cross_attention_dropout: float = 0.0,
        cross_attention_gate_init: float = 0.1,
    ):
        super().__init__()
        self.rna_dim = int(rna_dim)
        self.img_channels = int(img_channels)
        self.img_size = int(img_size)
        self.rna_embed_dim = int(model_channels * 4)

        self.rna_encoder = ExpressionRoutedGeneTokenEncoderV2(
            num_genes=rna_dim,
            token_dim=gene_token_dim,
            output_dim=self.rna_embed_dim,
            num_layers=gene_transformer_layers,
            num_heads=gene_transformer_heads,
            ff_dim=gene_transformer_ff_dim,
            dropout=gene_transformer_dropout,
            max_cross_attention_genes=max_cross_attention_genes,
            cross_attention_expression_threshold=(
                cross_attention_expression_threshold
            ),
            routing_mode=routing_mode,
            soft_routing_temperature=soft_routing_temperature,
        )
        self.unet = RNACrossAttentionUNet(
            in_channels=img_channels,
            model_channels=model_channels,
            out_channels=img_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            rna_embed_dim=self.rna_embed_dim,
            rna_strength=rna_strength,
            gene_token_dim=gene_token_dim,
            cross_attention_resolutions=cross_attention_resolutions,
            cross_attention_heads=cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dropout=cross_attention_dropout,
            cross_attention_gate_init=cross_attention_gate_init,
        )

    def set_routing_mode(
        self,
        routing_mode: str,
        soft_routing_temperature: float | None = None,
    ) -> None:
        self.rna_encoder.set_routing_mode(
            routing_mode,
            soft_routing_temperature=soft_routing_temperature,
        )

    @staticmethod
    def is_conditioning_parameter_name(name: str) -> bool:

        return (
            name.startswith("rna_encoder.")
            or name.startswith("unet.rna_proj.")
            or "rna_adapter" in name
            or "cross_attention" in name
        )

    def named_conditioning_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and self.is_conditioning_parameter_name(name):
                yield name, parameter

    def conditioning_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_conditioning_parameters():
            yield parameter

    def cross_attention_gate_values(self) -> torch.Tensor:
        gates = [
            torch.sigmoid(module.gate_logit)
            for module in self.unet.modules()
            if isinstance(module, SpatialGeneCrossAttention)
        ]
        if not gates:
            raise RuntimeError("GeneFlowV2 has no spatial gene cross-attention gates")
        return torch.stack(gates)

    def encode_rna(
        self,
        gene_expr: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
        condition_present: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        return self.rna_encoder(
            gene_expr,
            gene_mask=gene_mask,
            condition_present=condition_present,
            return_attention=return_attention,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        gene_expr: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
        condition_present: Optional[torch.Tensor] = None,
        return_cross_attention: bool = False,
    ):
        conditioning = self.rna_encoder.encode_conditioning(
            gene_expr,
            gene_mask=gene_mask,
            condition_present=condition_present,
            return_attention=False,
        )
        result = self.unet(
            x,
            t,
            extra={
                "rna_embedding": conditioning.global_embedding,
                "gene_tokens": conditioning.spatial_tokens,
                "gene_token_mask": conditioning.spatial_token_mask,
            },
            return_cross_attention=return_cross_attention,
        )
        if not return_cross_attention:
            return result
        output, token_attention = result
        return (
            output,
            token_attention[:, 0],
            token_attention[:, 1:],
            conditioning.active_gene_mask,
        )

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        gene_expr: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return self.forward(x, t, gene_expr, gene_mask=gene_mask)

        batch_size = gene_expr.shape[0]
        conditional = torch.ones(batch_size, dtype=torch.bool, device=gene_expr.device)
        unconditional = torch.zeros(batch_size, dtype=torch.bool, device=gene_expr.device)
        v_cond = self.forward(
            x,
            t,
            gene_expr,
            gene_mask=gene_mask,
            condition_present=conditional,
        )
        v_uncond = self.forward(
            x,
            t,
            gene_expr,
            gene_mask=gene_mask,
            condition_present=unconditional,
        )
        return v_uncond + guidance_scale * (v_cond - v_uncond)


def geneflowv2_config_from_args(args, rna_dim: int) -> dict:
    return {
        "rna_dim": int(rna_dim),
        "img_channels": args.img_channels,
        "img_size": args.img_size,
        "model_channels": args.model_channels,
        "num_res_blocks": args.num_res_blocks,
        "attention_resolutions": tuple(args.attention_resolutions),
        "dropout": args.dropout,
        "channel_mult": tuple(args.channel_mult),
        "use_checkpoint": args.use_checkpoint,
        "num_heads": args.num_heads,
        "num_head_channels": args.num_head_channels,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "rna_strength": args.rna_strength,
        "gene_token_dim": args.gene_token_dim,
        "gene_transformer_layers": args.gene_transformer_layers,
        "gene_transformer_heads": args.gene_transformer_heads,
        "gene_transformer_ff_dim": args.gene_transformer_ff_dim,
        "gene_transformer_dropout": args.gene_transformer_dropout,
        "max_cross_attention_genes": args.max_cross_attention_genes,
        "cross_attention_expression_threshold": (
            args.cross_attention_expression_threshold
        ),
        "routing_mode": getattr(args, "routing_mode", "hard"),
        "soft_routing_temperature": getattr(
            args, "soft_routing_temperature", 1.0
        ),
        "cross_attention_resolutions": tuple(args.cross_attention_resolutions),
        "cross_attention_heads": args.cross_attention_heads,
        "cross_attention_head_dim": args.cross_attention_head_dim,
        "cross_attention_dropout": args.cross_attention_dropout,
        "cross_attention_gate_init": args.cross_attention_gate_init,
    }
