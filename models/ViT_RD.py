import torch
from torch import Tensor, nn


def mask_from_bags(
    bags: Tensor,
    bag_sizes: Tensor,
) -> Tensor:
    max_possible_bag_size = bags.size(1)
    mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(0).repeat(
        bag_sizes.shape[0], 1
    ) >= bag_sizes.unsqueeze(1)

    return mask

class RunningMeanScaler(nn.Module):
    """Scales values by the inverse of the mean of values seen before."""

    def __init__(self, dtype=torch.float32) -> None:
        super().__init__()
        self.register_buffer("running_mean", torch.ones(1, dtype=dtype))
        self.register_buffer("items_so_far", torch.ones(1, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean_x = (self.running_mean + (x - self.running_mean) / self.items_so_far).mean()
            self.running_mean.copy_(mean_x)
            self.items_so_far += 1
        return x / self.running_mean


class ALiBi(nn.Module):
    """Per-head ALiBi mechanism with learned distance scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_distance = RunningMeanScaler()
        self.bias_scale = nn.Parameter(torch.rand(1))

    def forward(
        self,
        q: Tensor,  # (batch, query, qk_feature)
        k: Tensor,  # (batch, key, qk_feature)
        v: Tensor,  # (batch, key, v_feature)
        coords_q: Tensor,  # (batch, query, coord)
        coords_k: Tensor,  # (batch, key, coord)
        attn_mask: Tensor,  # (batch, query, key), bool
        alibi_mask: Tensor,  # (batch, query, key), bool
    ) -> Tensor:  # (batch, query, v_feature)

        weight_logits = torch.einsum("bqf,bkf->bqk", q, k) * (k.size(-1) ** -0.5)
        distances = torch.linalg.norm(coords_q.unsqueeze(2) - coords_k.unsqueeze(1), dim=-1)
        scaled_distances = self.scale_distance(distances) * self.bias_scale
        masked_distances = torch.where(alibi_mask, torch.zeros_like(scaled_distances), scaled_distances)

        weights = torch.softmax(weight_logits, dim=-1)
        masked = torch.where(attn_mask, torch.zeros_like(weights), weights - masked_distances)

        attention = torch.einsum("bqk,bkf->bqf", masked, v)
        return attention


class MultiHeadALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for multi-head attention.

    Reference:
    Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    """

    def __init__(self, *, embed_dim: int, num_heads: int) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"{embed_dim=} has to be divisible by {num_heads=}")

        self.query_encoders = nn.ModuleList(
            [
                nn.Linear(in_features=embed_dim, out_features=embed_dim // num_heads)
                for _ in range(num_heads)
            ]
        )
        self.key_encoders = nn.ModuleList(
            [
                nn.Linear(in_features=embed_dim, out_features=embed_dim // num_heads)
                for _ in range(num_heads)
            ]
        )
        self.value_encoders = nn.ModuleList(
            [
                nn.Linear(in_features=embed_dim, out_features=embed_dim // num_heads)
                for _ in range(num_heads)
            ]
        )

        self.attentions = nn.ModuleList([_ALiBi() for _ in range(num_heads)])

        self.fc = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(
        self,
        q: Tensor,        # (batch, query, embed_dim)
        k: Tensor,        # (batch, key, embed_dim)
        v: Tensor,        # (batch, key, embed_dim)
        coords_q: Tensor, # (batch, query, coord_dim)
        coords_k: Tensor, # (batch, key, coord_dim)
        attn_mask: Tensor,  # (batch, query, key), bool
        alibi_mask: Tensor  # (batch, query, key), bool
    ) -> Tensor:         # (batch, query, embed_dim)

        stacked_attentions = torch.stack(
            [
                att(
                    q=q_enc(q),
                    k=k_enc(k),
                    v=v_enc(v),
                    coords_q=coords_q,
                    coords_k=coords_k,
                    attn_mask=attn_mask,
                    alibi_mask=alibi_mask,
                )
                for q_enc, k_enc, v_enc, att in zip(
                    self.query_encoders,
                    self.key_encoders,
                    self.value_encoders,
                    self.attentions,
                    strict=True,
                )
            ]
        )
        return self.fc(stacked_attentions.permute(1, 2, 0, 3).flatten(-2, -1))

from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor, nn
from einops import repeat

# from stamp.modeling.alibi import MultiHeadALiBi


def feed_forward(
    dim: int,
    hidden_dim: int,
    dropout: float = 0.5,
) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class _RunningMeanScaler(nn.Module):
    """Scales values by the inverse of the mean of values seen before."""

    def __init__(self, dtype=torch.float32) -> None:
        super().__init__()
        self.running_mean = nn.parameter.Buffer(torch.ones(1, dtype=dtype))
        self.items_so_far = nn.parameter.Buffer(torch.ones(1, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Welford's algorithm
            self.running_mean.copy_(
                (self.running_mean + (x - self.running_mean) / self.items_so_far).mean()
            )
            self.items_so_far += 1

        return x / self.running_mean

class _ALiBi(nn.Module):
    # See MultiHeadAliBi
    def __init__(self) -> None:
        super().__init__()

        self.scale_distance = _RunningMeanScaler()
        self.bias_scale = nn.Parameter(torch.rand(1))

    def forward(
        self,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        coords_q: Tensor,
        coords_k: Tensor,
        attn_mask: Optional[Tensor],
        alibi_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            alibi_mask:
                Which query-key pairs to mask from ALiBi (i.e. don't apply ALiBi to).
        """
        weight_logits = torch.einsum("bqf,bkf->bqk", q, k) * (k.size(-1) ** -0.5)
        distances = torch.linalg.norm(
            coords_q.unsqueeze(2) - coords_k.unsqueeze(1), dim=-1
        )
        scaled_distances = self.scale_distance(distances) * self.bias_scale
        masked_distances = scaled_distances.where(~alibi_mask, 0.0)

        weights = torch.softmax(weight_logits, dim=-1)
        masked = (weights - masked_distances).where(~attn_mask, 0.0)

        attention = torch.einsum("bqk,bvf->bqf", masked, v)

        return attention
    
class SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.norm = nn.LayerNorm(dim)

        if use_alibi:
            self.mhsa = MultiHeadALiBi(
                embed_dim=dim,
                num_heads=num_heads,
            )
        else:
            self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        x: Tensor,
        *,
        coords: Tensor,
        attn_mask: Optional[Tensor],
        alibi_mask: Tensor,
    ) -> Tensor:
        x = self.norm(x)
        match self.mhsa:
            case nn.MultiheadAttention():
                attn_output, _ = self.mhsa(
                    x,
                    x,
                    x,
                    need_weights=False,
                    attn_mask=(
                        attn_mask.repeat(self.mhsa.num_heads, 1, 1)
                        if attn_mask is not None
                        else None
                    ),
                )
            case MultiHeadALiBi():
                attn_output = self.mhsa(
                    q=x,
                    k=x,
                    v=x,
                    coords_q=coords,
                    coords_k=coords,
                    attn_mask=attn_mask,
                    alibi_mask=alibi_mask,
                )
            case _:
                raise ValueError("Unexpected attention module")

        return attn_output

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.SelfAttention = SelfAttention(
                            dim=dim,
                            num_heads=heads,
                            dropout=dropout,
                            use_alibi=use_alibi,
                        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self.SelfAttention,
                        feed_forward(
                            dim,
                            mlp_dim,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        *,
        coords: Tensor,
        attn_mask: Optional[Tensor],
        alibi_mask: Tensor,
    ) -> Tensor:
        for attn, ff in self.layers:
            x_attn = attn(x, coords=coords, attn_mask=attn_mask, alibi_mask=alibi_mask)
            x = x_attn + x
            x = ff(x) + x

        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim_output: int,
        dim_input: int,
        dim_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(dim_model))

        self.project_features = nn.Sequential(
            nn.Linear(dim_input, dim_model, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.transformer = Transformer(
            dim=dim_model,
            depth=n_layers,
            heads=n_heads,
            mlp_dim=dim_feedforward,
            dropout=dropout,
            use_alibi=use_alibi,
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim_model, dim_output))

    def forward(
        self,
        bags: Tensor,
        *,
        coords: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, _n_tiles, _n_features = bags.shape

        bags = self.project_features(bags)

        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        bags = torch.cat([cls_tokens, bags], dim=1)
        coords = torch.cat([
            torch.zeros(batch_size, 1, 2).type_as(coords),
            coords
        ], dim=1)

        if mask is None:
            bags = self.transformer(bags, coords=coords, attn_mask=None, alibi_mask=torch.zeros(bags.shape[0], bags.shape[1], bags.shape[1], dtype=torch.bool, device=bags.device))
        else:
            mask_with_class_token = torch.cat(
                [torch.zeros(mask.shape[0], 1).type_as(mask), mask], dim=1
            )
            square_attn_mask = torch.einsum(
                "bq,bk->bqk", mask_with_class_token, mask_with_class_token
            )
            square_attn_mask[:, 1:, 0] = True

            alibi_mask = torch.zeros_like(square_attn_mask)
            alibi_mask[:, 0, :] = True
            alibi_mask[:, :, 0] = True

            bags = self.transformer(
                bags,
                coords=coords,
                attn_mask=square_attn_mask,
                alibi_mask=alibi_mask,
            )

        return self.mlp_head(bags[:, 0])
    
# vision_transformer = VisionTransformer(
#             dim_output=2,
#             dim_input=1024,
#             dim_model=192,
#             n_layers=2,
#             n_heads=4,
#             dim_feedforward=768,
#             dropout=0.1,
#             use_alibi=True,
#         )
# from typing import BinaryIO, Generic, NewType, TextIO, TypeAlias, TypeVar, cast
# # from jaxtyping import Bool, Float, Integer


# bags = torch.rand(1, 16, 1024)
# coords = torch.rand(1, 16, 2)
# bagSizes = torch.rand(16)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Move the model to GPU
# vision_transformer = vision_transformer.to(device)

# # Move input tensors to GPU
# bags = bags.to(device)
# coords = coords.to(device)
# bagSizes = bagSizes.to(device)

# # Generate the mask on GPU
# mask = mask_from_bags(bags=bags, bag_sizes=bagSizes).to(device)

# # Forward pass
# logits = vision_transformer(bags, coords=coords, mask=mask)

# print(logits)