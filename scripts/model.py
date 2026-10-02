"""Multi-head CNN: predicts s, mu, and intro generation from the r1/r2 AFS.

Backbones are interchangeable (they only differ in how the spectra are encoded);
they all feed the same FC + 3-head structure.
"""

import numpy as np
import torch
import torch.nn as nn

# s is mapped to an unbounded logit z via a logistic transform over [-delta, S_MAX+delta].
S_MAX = 2.5
DELTA = 1e-3
N_INTRO = 10  # intro_generation classes 0..9


def s_to_z(s, S_max=S_MAX, delta=DELTA):
    """Selection coefficient -> logit target (torch)."""
    t = (s + delta) / (S_max + 2 * delta)
    t = t.clamp(1e-6, 1 - 1e-6)
    return torch.log(t) - torch.log1p(-t)


def z_to_s(z, S_max=S_MAX, delta=DELTA):
    """Logit prediction -> selection coefficient (numpy)."""
    t = 1.0 / (1.0 + np.exp(-z))
    return t * (S_max + 2 * delta) - delta


def make_conv_block_1d(in_channels, out_channels, kernel_size=5, pool=2):
    """Conv1d (same padding) -> BN -> ReLU -> MaxPool. Halves length at pool=2."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(pool),
    )


class CrossAttention(nn.Module):
    """Single cross-attention block: query attends over key/value sequence."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, q_in, kv_in):
        out, _ = self.attn(self.query(q_in), self.key(kv_in), self.value(kv_in))
        return out


class _BackboneBase(nn.Module):
    """Shared aux-feature handling. Subclasses pass their flattened feature dim
    to super().__init__ and call _finish() with the encoded spectra. cn and the
    6 read totals are each embedded to hidden_dim//2 before the shared layer."""

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.cn_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.ReLU(),
        )
        self.tot_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.ReLU(),
        )
        # feat_dim + (hidden//2 from cn) + (hidden//2 from tot) = feat_dim + hidden_dim
        self.shared = nn.Sequential(nn.Linear(feat_dim + hidden_dim, hidden_dim), nn.ReLU())

    def _finish(self, spec_feats, X_cn, X_tot):
        B = spec_feats.size(0)
        cn = torch.log2(X_cn.view(B, 1).clamp_min(1e-6))
        h = torch.cat([spec_feats, self.cn_mlp(cn), self.tot_mlp(X_tot)], dim=1)
        return self.shared(h)


class BackboneCNNAttn(_BackboneBase):
    """One conv block per channel, then bidirectional cross-attention r1<->r2."""

    def __init__(self, hidden_dim=128, num_filters=32, num_heads=4):
        super().__init__(num_filters * 50 * 2, hidden_dim)  # 50 = length after one pool(2)
        self.conv_r1 = make_conv_block_1d(1, num_filters)
        self.conv_r2 = make_conv_block_1d(1, num_filters)
        self.cross_r1 = CrossAttention(num_filters, num_heads)
        self.cross_r2 = CrossAttention(num_filters, num_heads)

    def forward(self, X_r1, X_r2, X_cn, X_tot):
        B = X_r1.size(0)
        f1 = self.conv_r1(X_r1.unsqueeze(1)).transpose(1, 2)  # (B,50,F)
        f2 = self.conv_r2(X_r2.unsqueeze(1)).transpose(1, 2)
        a1 = self.cross_r1(f1, f2).reshape(B, -1)
        a2 = self.cross_r2(f2, f1).reshape(B, -1)
        return self._finish(torch.cat([a1, a2], dim=1), X_cn, X_tot)


class BackboneCNNNoAttn(_BackboneBase):
    """Same conv blocks as the attention model, but flatten raw features directly."""

    def __init__(self, hidden_dim=128, num_filters=32, **kw):
        super().__init__(num_filters * 50 * 2, hidden_dim)
        self.conv_r1 = make_conv_block_1d(1, num_filters)
        self.conv_r2 = make_conv_block_1d(1, num_filters)

    def forward(self, X_r1, X_r2, X_cn, X_tot):
        B = X_r1.size(0)
        f1 = self.conv_r1(X_r1.unsqueeze(1)).reshape(B, -1)  # (B, F*50)
        f2 = self.conv_r2(X_r2.unsqueeze(1)).reshape(B, -1)
        return self._finish(torch.cat([f1, f2], dim=1), X_cn, X_tot)


class BackboneR1Only(_BackboneBase):
    """Process only the r1 channel; r2 is ignored."""

    def __init__(self, hidden_dim=128, num_filters=32, **kw):
        super().__init__(num_filters * 50, hidden_dim)
        self.conv_r1 = make_conv_block_1d(1, num_filters)

    def forward(self, X_r1, X_r2, X_cn, X_tot):
        B = X_r1.size(0)
        f1 = self.conv_r1(X_r1.unsqueeze(1)).reshape(B, -1)
        return self._finish(f1, X_cn, X_tot)


class BackboneCNN2Ch(_BackboneBase):
    """Stack r1/r2 as a 2-channel input, one shared encoder, global average pool."""

    def __init__(self, hidden_dim=128, num_filters=32, **kw):
        super().__init__(num_filters * 2, hidden_dim)  # 2nd block outputs num_filters*2 channels
        self.conv = nn.Sequential(
            make_conv_block_1d(2, num_filters),
            make_conv_block_1d(num_filters, num_filters * 2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, X_r1, X_r2, X_cn, X_tot):
        x = torch.stack([X_r1, X_r2], dim=1)  # (B,2,100)
        f = self.global_pool(self.conv(x)).squeeze(-1)  # (B, 2F)
        return self._finish(f, X_cn, X_tot)


class BackboneCNNAttnPool(_BackboneBase):
    """Two conv blocks per channel, bidirectional cross-attention, then global
    average pool over positions (the pooled alternative to the flatten model)."""

    def __init__(self, hidden_dim=128, num_filters=32, num_heads=4):
        attn_dim = num_filters * 2          # channels after two conv blocks
        super().__init__(attn_dim * 2, hidden_dim)   # 2 streams, pooled to attn_dim each
        self.conv_r1 = nn.Sequential(make_conv_block_1d(1, num_filters),
                                     make_conv_block_1d(num_filters, attn_dim))
        self.conv_r2 = nn.Sequential(make_conv_block_1d(1, num_filters),
                                     make_conv_block_1d(num_filters, attn_dim))
        self.cross_r1 = CrossAttention(attn_dim, num_heads)
        self.cross_r2 = CrossAttention(attn_dim, num_heads)

    def forward(self, X_r1, X_r2, X_cn, X_tot):
        f1 = self.conv_r1(X_r1.unsqueeze(1)).transpose(1, 2)  # (B, 25, 2F)
        f2 = self.conv_r2(X_r2.unsqueeze(1)).transpose(1, 2)
        a1 = self.cross_r1(f1, f2).mean(dim=1)  # pool over positions -> (B, 2F)
        a2 = self.cross_r2(f2, f1).mean(dim=1)
        return self._finish(torch.cat([a1, a2], dim=1), X_cn, X_tot)


BACKBONES = {
    "cnn_attn": BackboneCNNAttn,
    "cnn_attn_pool": BackboneCNNAttnPool,
    "cnn_noattn": BackboneCNNNoAttn,
    "r1_only": BackboneR1Only,
    "cnn_2ch": BackboneCNN2Ch,
}


class MultiHeadModel(nn.Module):
    """Backbone (-> hidden_dim) + heads for s, mu, g. `heads` drops heads for the
    single-task ablations; `intro_mode` is softmax or an ordinal (n_intro-1) head."""

    def __init__(self, backbone, hidden_dim=128, n_intro=N_INTRO,
                 intro_mode="softmax", heads=("s", "mu", "g")):
        super().__init__()
        self.backbone = backbone
        self.intro_mode = intro_mode
        self.heads = tuple(heads)
        self.s_head = nn.Linear(hidden_dim, 1) if "s" in self.heads else None
        self.mu_head = nn.Linear(hidden_dim, 1) if "mu" in self.heads else None
        if "g" in self.heads:
            n_out = n_intro if intro_mode == "softmax" else n_intro - 1
            self.intro_head = nn.Linear(hidden_dim, n_out)
        else:
            self.intro_head = None

    def forward(self, X_r1, X_r2, X_cn, X_tot):
        h = self.backbone(X_r1, X_r2, X_cn, X_tot)
        out = {}
        if self.s_head is not None:
            out["s_raw"] = self.s_head(h).squeeze(-1)
        if self.mu_head is not None:
            out["mu_raw"] = torch.sigmoid(self.mu_head(h)).squeeze(-1)
        if self.intro_head is not None:
            out["intro_logits"] = self.intro_head(h)
        return out


def build_model(backbone="cnn_attn", hidden_dim=128, num_filters=32, num_heads=4,
                n_intro=N_INTRO, intro_mode="softmax", heads=("s", "mu", "g")):
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(BACKBONES)}")
    bb = BACKBONES[backbone](hidden_dim=hidden_dim, num_filters=num_filters, num_heads=num_heads)
    return MultiHeadModel(bb, hidden_dim=hidden_dim, n_intro=n_intro,
                          intro_mode=intro_mode, heads=heads)
