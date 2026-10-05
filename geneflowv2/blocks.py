from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

def conv_nd(dims, *args, **kwargs):

    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):

    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):

    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):

    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):

    return nn.GroupNorm(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):


    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def checkpoint(func, inputs, params, flag):


    if flag:


        del params
        return torch_checkpoint(func, *tuple(inputs), use_reentrant=False)
    else:
        return func(*inputs)

class TimestepBlock(nn.Module):

    def forward(self, x, emb):

        raise NotImplementedError()

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):


    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):


    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):


    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):


    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):


        return checkpoint(
            self._forward, 
            (x, emb), 
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):


    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, 
            (x,), 
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):


    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):


        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

def _rna_adapter(rna_embed_dim: int, time_embed_dim: int) -> nn.Module:

    return nn.Sequential(
        nn.SiLU(),
        linear(rna_embed_dim, time_embed_dim),
        nn.SiLU(),
        zero_module(linear(time_embed_dim, time_embed_dim)),
    )

class RNAFiLMUNet(nn.Module):


    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions,
        dropout: float = 0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        rna_embed_dim: int = 128,
        rna_strength: float = 1.0,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.model_channels = model_channels
        self.time_embed_dim = self.model_channels * 4
        self.rna_strength = rna_strength
        self._feature_size = 0

        self.time_embed = nn.Sequential(
            linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )


        self.rna_proj = nn.Sequential(
            linear(rna_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

        self.input_rna_adapters = nn.ModuleList(
            [_rna_adapter(rna_embed_dim, self.time_embed_dim) for _ in self.input_blocks]
        )
        self.middle_rna_adapter = _rna_adapter(rna_embed_dim, self.time_embed_dim)
        self.output_rna_adapters = nn.ModuleList(
            [_rna_adapter(rna_embed_dim, self.time_embed_dim) for _ in self.output_blocks]
        )

    def _conditioned_emb(self, t_emb, rna_emb, adapter):
        global_rna_emb = self.rna_proj(rna_emb)
        block_rna_emb = adapter(rna_emb)
        return t_emb + self.rna_strength * (global_rna_emb + block_rna_emb)

    def forward(self, x, timesteps, extra):
        if "rna_embedding" not in extra:
            raise ValueError("GeneFlowV2 requires extra['rna_embedding']")

        rna_emb = extra["rna_embedding"]
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        h = x
        for module, adapter in zip(self.input_blocks, self.input_rna_adapters):
            h = module(h, self._conditioned_emb(t_emb, rna_emb, adapter))
            hs.append(h)

        h = self.middle_block(
            h, self._conditioned_emb(t_emb, rna_emb, self.middle_rna_adapter)
        )

        for module, adapter in zip(self.output_blocks, self.output_rna_adapters):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, self._conditioned_emb(t_emb, rna_emb, adapter))

        h = h.type(x.dtype)
        return self.out(h)

@dataclass
class RNAConditioning:


    global_embedding: torch.Tensor
    gene_tokens: torch.Tensor
    gene_token_mask: torch.Tensor
    pooling_attention: Optional[torch.Tensor] = None

class GeneTokenTransformerEncoder(nn.Module):


    def __init__(
        self,
        num_genes: int,
        token_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {num_genes}")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_genes = int(num_genes)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)

        self.gene_identity = nn.Parameter(torch.empty(num_genes, token_dim))
        self.expression_encoder = nn.Sequential(
            nn.Linear(1, token_dim // 2),
            nn.SiLU(),
            nn.Linear(token_dim // 2, token_dim),
        )
        self.observation_embedding = nn.Embedding(2, token_dim)
        self.input_norm = nn.LayerNorm(token_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )


        self.pool_query = nn.Parameter(torch.empty(1, 1, token_dim))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )


        self.null_embedding = nn.Parameter(torch.empty(1, output_dim))
        self.null_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gene_identity, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)
        nn.init.normal_(self.null_embedding, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)

    def _validate_inputs(
        self,
        expression: torch.Tensor,
        gene_mask: Optional[torch.Tensor],
        condition_present: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if expression.ndim != 2 or expression.shape[1] != self.num_genes:
            raise ValueError(
                f"expression must have shape [batch, {self.num_genes}], "
                f"got {tuple(expression.shape)}"
            )
        if gene_mask is None:
            observed = torch.ones_like(expression, dtype=torch.bool)
        else:
            if gene_mask.shape != expression.shape:
                raise ValueError(
                    f"gene_mask shape {tuple(gene_mask.shape)} does not match "
                    f"expression shape {tuple(expression.shape)}"
                )
            observed = gene_mask.to(device=expression.device) > 0

        batch_size = expression.shape[0]
        if condition_present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=expression.device)
        else:
            present = condition_present.to(device=expression.device, dtype=torch.bool)
            if present.ndim == 0:
                present = present.expand(batch_size)
            if present.shape != (batch_size,):
                raise ValueError(
                    f"condition_present must have shape [{batch_size}], "
                    f"got {tuple(present.shape)}"
                )
        return observed, present

    def encode_conditioning(
        self,
        expression: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
        condition_present: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> RNAConditioning:
        observed, present = self._validate_inputs(
            expression, gene_mask, condition_present
        )
        batch_size = expression.shape[0]


        clean_expression = torch.nan_to_num(expression) * observed.to(expression.dtype)
        value_tokens = self.expression_encoder(clean_expression.unsqueeze(-1))
        gene_tokens = self.gene_identity.unsqueeze(0).expand(batch_size, -1, -1)
        observation_tokens = self.observation_embedding(observed.long())
        tokens = self.input_dropout(
            self.input_norm(gene_tokens + value_tokens + observation_tokens)
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
        null_embedding = self.null_embedding.expand(batch_size, -1)
        global_embedding = torch.where(
            effective_present[:, None], conditional_embedding, null_embedding
        )

        null_tokens = self.null_token.expand(batch_size, self.num_genes, -1)
        conditioned_tokens = torch.where(
            effective_present[:, None, None], encoded_tokens, null_tokens
        )
        null_mask = torch.zeros_like(observed)
        null_mask[:, 0] = True
        conditioned_mask = torch.where(
            effective_present[:, None], observed, null_mask
        )

        pooling_attention = None
        if return_attention:

            pooling_attention = attention[:, :, 0].mean(dim=1)
            pooling_attention = pooling_attention.masked_fill(~observed, 0.0)
            pooling_attention = torch.where(
                effective_present[:, None],
                pooling_attention,
                torch.zeros_like(pooling_attention),
            )

        return RNAConditioning(
            global_embedding=global_embedding,
            gene_tokens=conditioned_tokens,
            gene_token_mask=conditioned_mask,
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
            conditioning.gene_tokens,
            conditioning.pooling_attention,
        )

class SpatialGeneCrossAttention(nn.Module):


    def __init__(
        self,
        channels: int,
        token_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        gate_init: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if channels <= 0 or token_dim <= 0:
            raise ValueError("channels and token_dim must be positive")
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between zero and one")

        self.channels = int(channels)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.use_checkpoint = bool(use_checkpoint)


        self.feature_norm = nn.GroupNorm(32, channels)
        self.query_projection = nn.Conv1d(channels, self.inner_dim, 1)
        self.token_norm = nn.LayerNorm(token_dim)
        self.key_projection = nn.Linear(token_dim, self.inner_dim)
        self.value_projection = nn.Linear(token_dim, self.inner_dim)
        self.output_projection = nn.Conv1d(self.inner_dim, channels, 1)

        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))


        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _project(self, x, gene_tokens):
        batch_size, _, *spatial = x.shape
        spatial_count = math.prod(spatial)
        query = self.query_projection(self.feature_norm(x).flatten(2)).transpose(1, 2)
        normalized_tokens = self.token_norm(gene_tokens)
        key = self.key_projection(normalized_tokens)
        value = self.value_projection(normalized_tokens)

        def split_heads(tensor, length):
            return tensor.view(
                batch_size, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        return (
            split_heads(query, spatial_count),
            split_heads(key, gene_tokens.shape[1]),
            split_heads(value, gene_tokens.shape[1]),
            spatial,
        )

    def _attention(
        self,
        x: torch.Tensor,
        gene_tokens: torch.Tensor,
        gene_token_mask: torch.Tensor,
        return_attention: bool,
    ):
        if gene_tokens.ndim != 3 or gene_tokens.shape[0] != x.shape[0]:
            raise ValueError("gene_tokens must have shape [batch, genes, token_dim]")
        if gene_tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"expected token_dim={self.token_dim}, got {gene_tokens.shape[2]}"
            )
        if gene_token_mask.shape != gene_tokens.shape[:2]:
            raise ValueError("gene_token_mask must have shape [batch, genes]")
        if gene_token_mask.dtype == torch.bool:
            token_weights = None
            valid = gene_token_mask.to(device=x.device)
        else:


            token_weights = torch.nan_to_num(
                gene_token_mask.to(device=x.device, dtype=torch.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(min=0.0, max=1.0)
            valid = token_weights > 0
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample must expose at least one gene or null token")

        query, key, value, spatial = self._project(x, gene_tokens)
        attention_bias = torch.zeros(
            x.shape[0], 1, 1, gene_tokens.shape[1], device=x.device, dtype=query.dtype
        )
        if token_weights is not None:
            smallest = torch.finfo(query.dtype).tiny
            attention_bias = attention_bias + token_weights.clamp_min(
                smallest
            ).log()[:, None, None, :].to(query.dtype)
        attention_bias = attention_bias.masked_fill(
            ~valid[:, None, None, :], torch.finfo(query.dtype).min
        )
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).reshape(
            x.shape[0], math.prod(spatial), self.inner_dim
        )
        residual = self.output_projection(context.transpose(1, 2)).reshape(
            x.shape[0], self.channels, *spatial
        )
        output = x + torch.sigmoid(self.gate_logit).to(x.dtype) * residual

        if not return_attention:
            return output, None


        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        if token_weights is not None:
            scores = scores + token_weights.clamp_min(
                torch.finfo(scores.dtype).tiny
            ).log()[:, None, None, :]
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        gene_importance = weights.mean(dim=(1, 2)).to(x.dtype)
        gene_importance = gene_importance.masked_fill(~valid, 0.0)
        return output, gene_importance

    def forward(
        self,
        x: torch.Tensor,
        gene_tokens: torch.Tensor,
        gene_token_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        if self.use_checkpoint and self.training and not return_attention:
            return torch_checkpoint(
                lambda image, tokens, mask: self._attention(
                    image, tokens, mask, False
                )[0],
                x,
                gene_tokens,
                gene_token_mask,
                use_reentrant=False,
            )
        output, attention = self._attention(
            x, gene_tokens, gene_token_mask, return_attention
        )
        return (output, attention) if return_attention else output

class RNACrossAttentionUNet(RNAFiLMUNet):


    def __init__(
        self,
        *args,
        gene_token_dim: int,
        cross_attention_resolutions=(8, 16),
        cross_attention_heads: int = 8,
        cross_attention_head_dim: int = 32,
        cross_attention_dropout: float = 0.0,
        cross_attention_gate_init: float = 0.1,
        **kwargs,
    ):

        model_channels = int(kwargs.get("model_channels", args[1] if len(args) > 1 else 0))
        num_res_blocks = int(
            kwargs.get("num_res_blocks", args[3] if len(args) > 3 else 0)
        )
        channel_mult = tuple(kwargs.get("channel_mult", (1, 2, 4, 8)))
        use_checkpoint = bool(kwargs.get("use_checkpoint", False))
        super().__init__(*args, **kwargs)

        requested = tuple(sorted({int(value) for value in cross_attention_resolutions}))
        if not requested or any(value <= 0 for value in requested):
            raise ValueError("cross_attention_resolutions must contain positive factors")
        self.cross_attention_resolutions = requested

        input_specs = [(model_channels, 1)]
        input_channels = [model_channels]
        channels = model_channels
        downsample = 1
        for level, multiplier in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                channels = int(multiplier * model_channels)
                input_specs.append((channels, downsample))
                input_channels.append(channels)
            if level != len(channel_mult) - 1:
                input_specs.append((channels, downsample * 2))
                input_channels.append(channels)
                downsample *= 2

        middle_spec = (channels, downsample)
        output_specs = []
        for level, multiplier in list(enumerate(channel_mult))[::-1]:
            for block_index in range(num_res_blocks + 1):
                input_channels.pop()
                channels = int(multiplier * model_channels)
                if level and block_index == num_res_blocks:
                    downsample //= 2
                output_specs.append((channels, downsample))

        if len(input_specs) != len(self.input_blocks):
            raise RuntimeError("V2 input cross-attention layout does not match the UNet")
        if len(output_specs) != len(self.output_blocks):
            raise RuntimeError("V2 output cross-attention layout does not match the UNet")

        def make_block(spec):
            block_channels, factor = spec
            if factor not in self.cross_attention_resolutions:
                return nn.Identity()
            return SpatialGeneCrossAttention(
                channels=block_channels,
                token_dim=gene_token_dim,
                num_heads=cross_attention_heads,
                head_dim=cross_attention_head_dim,
                dropout=cross_attention_dropout,
                gate_init=cross_attention_gate_init,
                use_checkpoint=use_checkpoint,
            )

        self.input_cross_attention = nn.ModuleList(
            [make_block(spec) for spec in input_specs]
        )
        self.middle_cross_attention = make_block(middle_spec)
        self.output_cross_attention = nn.ModuleList(
            [make_block(spec) for spec in output_specs]
        )
        self.cross_attention_block_count = sum(
            isinstance(module, SpatialGeneCrossAttention)
            for module in (
                list(self.input_cross_attention)
                + [self.middle_cross_attention]
                + list(self.output_cross_attention)
            )
        )
        if self.cross_attention_block_count == 0:
            raise ValueError(
                "no cross-attention block matches the configured UNet resolutions"
            )

    @staticmethod
    def _apply_cross_attention(
        module,
        image,
        gene_tokens,
        gene_token_mask,
        return_attention,
    ):
        if not isinstance(module, SpatialGeneCrossAttention):
            return image, None
        if return_attention:
            return module(
                image, gene_tokens, gene_token_mask, return_attention=True
            )
        return module(image, gene_tokens, gene_token_mask), None

    def forward(self, x, timesteps, extra, return_cross_attention: bool = False):
        required = {"rna_embedding", "gene_tokens", "gene_token_mask"}
        missing = sorted(required.difference(extra))
        if missing:
            raise ValueError(f"GeneFlowV2 UNet missing conditioning fields: {missing}")

        rna_embedding = extra["rna_embedding"]
        gene_tokens = extra["gene_tokens"]
        gene_token_mask = extra["gene_token_mask"]
        time_embedding = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )

        attention_summaries = []
        skip_connections = []
        hidden = x
        for module, adapter, cross_attention in zip(
            self.input_blocks,
            self.input_rna_adapters,
            self.input_cross_attention,
        ):
            hidden = module(
                hidden,
                self._conditioned_emb(time_embedding, rna_embedding, adapter),
            )
            hidden, summary = self._apply_cross_attention(
                cross_attention,
                hidden,
                gene_tokens,
                gene_token_mask,
                return_cross_attention,
            )
            if summary is not None:
                attention_summaries.append(summary)
            skip_connections.append(hidden)

        hidden = self.middle_block(
            hidden,
            self._conditioned_emb(
                time_embedding, rna_embedding, self.middle_rna_adapter
            ),
        )
        hidden, summary = self._apply_cross_attention(
            self.middle_cross_attention,
            hidden,
            gene_tokens,
            gene_token_mask,
            return_cross_attention,
        )
        if summary is not None:
            attention_summaries.append(summary)

        for module, adapter, cross_attention in zip(
            self.output_blocks,
            self.output_rna_adapters,
            self.output_cross_attention,
        ):
            hidden = torch.cat([hidden, skip_connections.pop()], dim=1)
            hidden = module(
                hidden,
                self._conditioned_emb(time_embedding, rna_embedding, adapter),
            )
            hidden, summary = self._apply_cross_attention(
                cross_attention,
                hidden,
                gene_tokens,
                gene_token_mask,
                return_cross_attention,
            )
            if summary is not None:
                attention_summaries.append(summary)

        output = self.out(hidden.type(x.dtype))
        if not return_cross_attention:
            return output
        if not attention_summaries:
            raise RuntimeError("cross-attention diagnostics requested but none were produced")
        aggregate = torch.stack(attention_summaries, dim=0).mean(dim=0)
        return output, aggregate
