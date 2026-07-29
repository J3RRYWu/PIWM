"""Alternative road-context perception head: per-frame CNN -> VQ (discrete latent)
-> Transformer over the temporal window -> upcoming curvature profile.

WHY THIS EXISTS
The journal draft's architecture section claims the driving experiments use the
conference paper's strongest configuration -- *extrinsic encoding with a discrete
(VQ) latent* plus a *Transformer physical encoder*. The shipped road-context
encoder (models/road_perception.py) is neither: it stacks the 15 frames as conv
channels and reads curvature off a single linear layer. That conference result was
established on fully observable benchmarks (CartPole / Lunar Lander), so carrying
it over to a partially observable driving task is an extrapolation, not a finding.
This module implements the claimed architecture so the two can be compared
head-to-head on the actual task, and whichever wins goes in the paper.

DESIGN (deliberately mirrors the claim, not the existing code)
  per-frame CNN     each of the FS frames is encoded on its own, so temporal
                    structure is NOT collapsed into conv channels
  VQ bottleneck     512-entry codebook, straight-through estimator; the discrete
                    latent is the regularizer the conference paper credits
  Transformer       2 layers of self-attention over the FS tokens + learned
                    positional embedding, then mean-pool
  linear head       -> n_offsets curvature values, in real 1/m units

INTERFACE PARITY: forward(x) takes (B, FS, 64, 64) and returns (B, n_offsets),
exactly like RoadContextEncoder, so eval/rollout code (compare_kappa_rollout.py,
FrenetDynamics.rollout_perceived) works unchanged. The VQ auxiliary loss is not
returned; it is stashed on .last_vq_loss for the training loop to add, and
.last_perplexity reports codebook usage (a collapsed codebook would make the
comparison meaningless, so it is monitored rather than assumed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lane_utils import LANE_FRAME_STACK


class VectorQuantizer(nn.Module):
    """VQ-VAE quantizer with a straight-through estimator.

    Returns the quantized tensor, the auxiliary loss
        ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
    and the codebook perplexity (effective number of codes in use).

    Two anti-collapse measures, both necessary here. A first attempt with a plain
    quantizer (uniform +-1/n_codes init, no revival) collapsed to ~12 of 512 codes
    and the auxiliary loss spiked 30x in the first epochs, because the codes start
    orders of magnitude smaller than the encoder outputs and every sample lands on
    a handful of them. With a collapsed codebook the discrete bottleneck destroys
    almost all information, so the resulting comparison measures the quantizer's
    training rather than the architecture:
      * data-dependent init -- the codebook is seeded from real encoder outputs on
        the first training batch, so codes start inside the data distribution;
      * dead-code revival   -- codes unused for `revive_after` steps are reset to
        random encoder outputs from the current batch.
    """

    def __init__(self, n_codes=512, dim=128, beta=0.25, revive_after=200):
        super().__init__()
        self.n_codes, self.dim, self.beta = n_codes, dim, beta
        self.revive_after = revive_after
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.normal_(0, 0.5)
        self.register_buffer("_inited", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_idle", torch.zeros(n_codes, dtype=torch.long))

    @torch.no_grad()
    def _data_init(self, z):
        """Seed the codebook from real encoder outputs (k-means-free, one batch)."""
        n = z.shape[0]
        pick = torch.randint(0, n, (self.n_codes,), device=z.device)
        self.codebook.weight.data.copy_(z[pick].detach())
        self._inited.fill_(True)

    @torch.no_grad()
    def _revive(self, z, idx):
        """Reset codes that have gone unused for too long to random encoder outputs."""
        used = torch.zeros(self.n_codes, dtype=torch.bool, device=z.device)
        used[idx.unique()] = True
        self._idle[used] = 0
        self._idle[~used] += 1
        dead = (self._idle > self.revive_after).nonzero(as_tuple=True)[0]
        if dead.numel():
            pick = torch.randint(0, z.shape[0], (dead.numel(),), device=z.device)
            self.codebook.weight.data[dead] = z[pick].detach()
            self._idle[dead] = 0

    def forward(self, z):
        """z: (N, dim) -> (z_q (N,dim), aux_loss scalar, perplexity scalar)"""
        if self.training and not bool(self._inited):
            self._data_init(z)

        # squared euclidean distance to every code
        d = (z.pow(2).sum(1, keepdim=True)
             - 2 * z @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))
        idx = d.argmin(1)
        e = self.codebook(idx)

        codebook_loss = F.mse_loss(e, z.detach())
        commit_loss = F.mse_loss(z, e.detach())
        aux = codebook_loss + self.beta * commit_loss

        z_q = z + (e - z).detach()          # straight-through

        with torch.no_grad():
            probs = torch.bincount(idx, minlength=self.n_codes).float()
            probs = probs / probs.sum().clamp(min=1)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        if self.training:
            self._revive(z, idx)
        return z_q, aux, perplexity


class RoadContextVQFormer(nn.Module):
    # Defaults are sized so the total parameter count lands near the shipped CNN
    # encoder (~1.76M). Losing on equal capacity is a statement about the
    # architecture; losing at half the capacity would not be.
    def __init__(self, n_offsets=10, frame_stack=LANE_FRAME_STACK,
                 d_model=160, n_codes=512, n_layers=3, n_heads=4, ff=512):
        super().__init__()
        self.frame_stack = frame_stack
        self.d_model = d_model

        # --- per-frame vision encoder (shared weights across the window) ---
        self.frame_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),   # 8x8
        )
        self.frame_proj = nn.Linear(64 * 8 * 8, d_model)

        # --- discrete bottleneck ---
        self.vq = VectorQuantizer(n_codes=n_codes, dim=d_model)

        # --- Transformer over the temporal window ---
        self.pos = nn.Parameter(torch.zeros(1, frame_stack, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=0.0, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.kappa_head = nn.Linear(d_model, n_offsets)

        self.last_vq_loss = torch.zeros(())
        self.last_perplexity = torch.zeros(())

    def forward(self, x):
        """x: (B, FS, 64, 64) -> (B, n_offsets) perceived curvature profile."""
        B, T = x.shape[0], x.shape[1]
        h = self.frame_conv(x.reshape(B * T, 1, x.shape[2], x.shape[3]))
        h = self.frame_proj(h.reshape(B * T, -1))               # (B*T, d)

        z_q, aux, perp = self.vq(h)
        self.last_vq_loss, self.last_perplexity = aux, perp

        z_q = z_q.reshape(B, T, self.d_model) + self.pos
        h = self.transformer(z_q)                               # (B,T,d)
        return self.kappa_head(self.norm(h.mean(1)))
