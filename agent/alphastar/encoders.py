"""
Observation encoders for AlphaStar.

Three streams mirror the paper:
    ScalarEncoder          – non-spatial game state (resources, supply, etc.)
    EntityEncoder          – Transformer over the entity list (units / buildings)
    SpatialEncoder         – ResBlock CNN tower over screen feature planes
    EntityFeatureExtractor – converts raw PySC2 feature_units into fixed-dim vectors
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agent.alphastar.constants import (
    SCALAR_DIM, SCALAR_HIDDEN_DIM,
    ENTITY_FEATURE_DIM, ENTITY_HIDDEN_DIM,
    NUM_TRANSFORMER_HEADS, NUM_TRANSFORMER_LAYERS,
    SPATIAL_CHANNELS, SPATIAL_SIZE, SPATIAL_HIDDEN_DIM,
    MAX_ENTITIES, MAX_UNIT_TYPES,
    UNIT_TYPE_EMBEDDING_DIM, RAW_ENTITY_NUMERIC_DIM,
)


class ResBlock(nn.Module):
    """
    Standard residual block: two 3×3 convolutions with GroupNorm.
    GroupNorm (groups=8) is preferred over BatchNorm for the small or
    variable batch sizes common in RL.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class ScalarEncoder(nn.Module):
    """
    Encodes scalar (non-spatial) game state: resources, supply counts, etc.

    Input  : (B, SCALAR_DIM)        raw player stats (can be large integers)
    Output : (B, SCALAR_HIDDEN_DIM) normalised, embedded representation

    Preprocessing:
        log1p is applied before the MLP to compress the large dynamic range
        of mineral / vespene counts (0–10 000+).  LayerNorm after each linear
        layer stabilises training.
    """

    def __init__(
        self,
        input_dim: int = SCALAR_DIM,
        hidden_dim: int = SCALAR_HIDDEN_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # log1p compresses large-range scalars (minerals, vespene, etc.)
        return self.net(torch.log1p(x))


class EntityEncoder(nn.Module):
    """
    Transformer-based encoder over the set of entities (units / buildings).

    Entities are treated as an *unordered set*: SC2 units have no natural
    sequence order, and the Transformer's self-attention is already
    permutation-equivariant, so adding positional encodings is not appropriate
    here.  The paper confirms the set-based treatment.

    Returns
    -------
    entity_embeddings : (B, N, hidden_dim)  per-entity representations
    embedded_entity   : (B, hidden_dim)     mask-aware mean-pooled summary
    """

    def __init__(
        self,
        feature_dim: int = ENTITY_FEATURE_DIM,
        hidden_dim: int = ENTITY_HIDDEN_DIM,
        num_heads: int = NUM_TRANSFORMER_HEADS,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        entities: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        entities         : (B, N, feature_dim)
        key_padding_mask : (B, N) bool – True marks padding positions to ignore
        """
        x = self.input_proj(entities)              # (B, N, hidden_dim)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)   # (B, N, 1)
            embedded_entity = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            embedded_entity = x.mean(dim=1)        # (B, hidden_dim)

        return x, embedded_entity


class SpatialEncoder(nn.Module):
    """
    ResBlock-based CNN encoder for spatial map features (screen / minimap).

    Architecture:
        stem  : 8 → 32 ch, Conv 4x4 stride-2 + GroupNorm    (64 → 32)
              : 32 → 64 ch, Conv 4x4 stride-2 + GroupNorm    (32 → 16)
        body  : 2 x ResBlock(64)                              (16 x 16)
        head  : 64 → 128, Conv 1x1 + GroupNorm               (16 x 16)  ← map_skip
        proj  : Flatten → Linear(128·16·16, hidden_dim)

    Input  : (B, SPATIAL_CHANNELS, H, W)
    Output
    -------
    map_skip         : (B, 128, 16, 16)  spatial feature map for TargetLocationHead
                        (always 16×16 via AdaptiveAvgPool, regardless of input resolution)
    embedded_spatial : (B, SPATIAL_HIDDEN_DIM) compressed vector for LSTM input
    """

    def __init__(
        self,
        in_channels: int = SPATIAL_CHANNELS,
        hidden_dim: int = SPATIAL_HIDDEN_DIM,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            # Pool to a fixed spatial size so the downstream Linear and
            # TargetLocationHead are resolution-independent
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(
            128 * (SPATIAL_SIZE // 4) * (SPATIAL_SIZE // 4),
            hidden_dim,
        )

    def forward(self, spatial: torch.Tensor):
        x = self.stem(spatial)          # (B, 64, H/4, W/4)
        x = self.body(x)                # (B, 64, H/4, W/4)
        map_skip = self.head(x)         # (B, 128, H/4, W/4)
        embedded_spatial = self.proj(self.flatten(map_skip))   # (B, hidden_dim)
        return map_skip, embedded_spatial


class EntityFeatureExtractor(nn.Module):
    """
    Converts raw per-unit data from PySC2's `feature_units` observation into
    fixed-dimensional entity feature vectors suitable for EntityEncoder.

    Raw numeric features (RAW_ENTITY_NUMERIC_DIM = 16, all in [0, 1]):
        0-3  alliance one-hot  (self / ally / neutral / enemy)
        4    health_ratio
        5    shield_ratio
        6    energy_ratio
        7    build_progress / 100
        8    is_selected
        9    is_blip
        10   is_powered
        11   x / SPATIAL_SIZE
        12   y / SPATIAL_SIZE
        13   weapon_cooldown / 32
        14   attack_upgrade_level / 3
        15   armor_upgrade_level / 3
    """

    def __init__(
        self,
        unit_type_dim: int = UNIT_TYPE_EMBEDDING_DIM,
        max_unit_types: int = MAX_UNIT_TYPES,
        numeric_dim: int = RAW_ENTITY_NUMERIC_DIM,
        out_dim: int = ENTITY_FEATURE_DIM,
    ):
        super().__init__()
        self.unit_type_emb = nn.Embedding(max_unit_types, unit_type_dim, padding_idx=0)
        self.proj = nn.Linear(unit_type_dim + numeric_dim, out_dim)

    def forward(
        self,
        unit_types: torch.Tensor,   # (B, N) int64
        numeric: torch.Tensor,      # (B, N, RAW_ENTITY_NUMERIC_DIM) float32
    ) -> torch.Tensor:
        """Returns (B, N, ENTITY_FEATURE_DIM)."""
        type_emb = self.unit_type_emb(unit_types)          # (B, N, unit_type_dim)
        combined = torch.cat([type_emb, numeric], dim=-1)  # (B, N, unit_type_dim+numeric_dim)
        return F.relu(self.proj(combined))                 # (B, N, out_dim)

    @staticmethod
    def extract_from_obs(obs, max_entities: int = MAX_ENTITIES, spatial_size: int = SPATIAL_SIZE):
        """
        Parse ``obs.observation['feature_units']`` into numpy arrays.

        Returns
        -------
        unit_types : (1, max_entities) int64
        numeric    : (1, max_entities, RAW_ENTITY_NUMERIC_DIM) float32
        mask       : (1, max_entities) bool  – True = padding slot
        """
        raw = obs.observation.get('feature_units', [])
        unit_types_list = []
        numeric_list = []

        for u in raw[:max_entities]:
            hp_ratio = u.health / max(u.health_max, 1)
            sh_ratio = u.shield / max(u.shield_max, 1)
            en_ratio = u.energy / max(u.energy_max, 1)
            alliance = int(u.alliance)

            numeric_list.append([
                float(alliance == 1),              # self
                float(alliance == 2),              # ally
                float(alliance == 3),              # neutral
                float(alliance == 4),              # enemy
                hp_ratio,
                sh_ratio,
                en_ratio,
                float(u.build_progress) / 100.0,
                float(u.is_selected),
                float(u.is_blip),
                float(u.is_powered),
                float(u.x) / spatial_size,
                float(u.y) / spatial_size,
                float(u.weapon_cooldown) / 32.0,
                float(u.attack_upgrade_level) / 3.0,
                float(u.armor_upgrade_level) / 3.0,
            ])
            unit_types_list.append(min(int(u.unit_type), MAX_UNIT_TYPES - 1))

        n = len(unit_types_list)
        pad = max_entities - n

        unit_types_arr = np.array(unit_types_list + [0] * pad, dtype=np.int64)
        numeric_arr = np.zeros((max_entities, RAW_ENTITY_NUMERIC_DIM), dtype=np.float32)
        if n > 0:
            numeric_arr[:n] = numeric_list
        mask_arr = np.zeros(max_entities, dtype=bool)
        mask_arr[n:] = True   # True = padding

        return (
            unit_types_arr[None],   # (1, max_entities)
            numeric_arr[None],      # (1, max_entities, RAW_ENTITY_NUMERIC_DIM)
            mask_arr[None],         # (1, max_entities)
        )
