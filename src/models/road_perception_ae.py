"""The conference version's autoencoding encoders, adapted to road context.

The conference formulation has a 2x2 design space -- intrinsic vs extrinsic
architecture, continuous vs discrete latent -- and reports extrinsic as strongest.
The driving encoder shipped in this repository is neither: it is a directly
supervised regressor from images to the curvature profile, with no reconstruction
objective anywhere. That is defensible (road context has privileged supervision at
training time, so nothing has to be learned through reconstruction) but it is NOT
the conference configuration, and the article should not claim to inherit its
ranking without having run it. These two classes are that configuration, so the
claim can be tested rather than asserted.

STRUCTURE FOLLOWS THE REFERENCE IMPLEMENTATION (PIWM-main/ex-conti, in-conti):
five stride-2 conv layers, a Gaussian latent with reparameterisation, a mirrored
deconv decoder with a sigmoid output, and a plain MLP physical head.

FOUR THINGS ARE DELIBERATELY NOT COPIED, because the reference targets a different
problem and copying them would import known defects:

  1. The reference encodes ONE 224x224 RGB frame. Road context cannot be read from
     one frame -- the whole task is inferring the geometry ahead from a short
     window -- so the vision encoder here runs PER FRAME over the 15-frame stack
     and the physical head aggregates, which is also what the paper describes
     ("the physical encoder aggregates a context window of k = 15 frames").
  2. Its images are 224x224; ours are 64x64, so the same five stride-2 layers land
     on 2x2 instead of 7x7.
  3. Its target is `state[:, :2]`, two scalars. Ours is a 10-point curvature
     profile over arc length.
  4. Its train/val split is `random_split` over frames. Adjacent frames share 14 of
     15 stack members, so that split leaks; this repository holds out whole
     episodes (see src/folds.py). Training scripts here must use fold_split.

The extrinsic stage-1 objective is reconstruction and KL ONLY, with no physical
supervision -- that separation is what makes it extrinsic, and it is the reason
`checkpoints/piwm_lane_v6_donkey/ae.tar` does not qualify: that encoder was trained
with car- and lane-state terms in its loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lane_utils import LANE_FRAME_STACK


class _FrameConv(nn.Module):
    """Five stride-2 convolutions, per the reference VAE. 64x64 -> 512 x 2 x 2."""

    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.out_dim = 512 * 2 * 2

    def forward(self, x):                       # (N,1,64,64) -> (N, out_dim)
        return self.net(x).flatten(1)


class _FrameDeconv(nn.Module):
    """Mirror of _FrameConv, sigmoid output to match the [0,1] frames."""

    def __init__(self, latent_dim, out_ch=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):                       # (N, latent) -> (N,1,64,64)
        return self.net(self.fc(z).view(-1, 512, 2, 2))


def _kl(mu, logvar):
    """Summed over latent dims, averaged over the batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()


class ExtrinsicVisionVAE(nn.Module):
    """Stage 1: a general-purpose vision VAE. Reconstruction and KL only.

    No physical supervision reaches this stage -- that is the definition of the
    extrinsic split, and the property that lets stage 2 be interpreted as
    extracting physics from a representation that was not built for it.
    Frames are encoded independently; the stack dimension is folded into the batch.
    """

    def __init__(self, latent_dim=128, frame_stack=LANE_FRAME_STACK):
        super().__init__()
        self.latent_dim, self.frame_stack = latent_dim, frame_stack
        self.enc = _FrameConv(1)
        self.fc_mu = nn.Linear(self.enc.out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc.out_dim, latent_dim)
        self.dec = _FrameDeconv(latent_dim, 1)

    def encode(self, x):
        """x: (B, FS, 64, 64) -> mu, logvar, each (B, FS, latent_dim)."""
        b, fs = x.shape[:2]
        h = self.enc(x.reshape(b * fs, 1, *x.shape[2:]))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu.view(b, fs, -1), logvar.view(b, fs, -1)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x):
        """Encode every frame (stage 2 reads all of them), reconstruct only the last.

        WHY ONLY THE LAST. Decoding all 15 frames means pushing b*15 images through
        the deconv stack; at batch 64 that is 960, and the 32x32x32 activation alone
        is ~126 MB before autograd keeps its intermediates -- measured at 4.6 GB per
        process, which is unusable at any real concurrency.

        Nothing is lost. The encoder is one shared per-frame function, not fifteen,
        and the dataset's windows slide by one frame, so every frame in the dataset
        is the last frame of exactly one window: an epoch still sees each frame once.
        What disappears is only the 15-fold redundant re-encoding of the same frames.

        The KL follows the reconstruction onto the same frame. Applying it to all
        fifteen while training only one would leave the other fourteen latents with
        nothing but a pull toward the prior, and stage 2 reads all fifteen.
        """
        mu, logvar = self.encode(x)
        mu_l, logvar_l = mu[:, -1], logvar[:, -1]
        rec = self.dec(self.reparameterize(mu_l, logvar_l))
        return rec.squeeze(1), mu, logvar

    def loss(self, x, kl_weight=1.0):
        rec, mu, logvar = self(x)
        tgt = x[:, -1]
        # BCE as in the reference; the frames are already in [0,1]
        recon = F.binary_cross_entropy(rec.clamp(1e-6, 1 - 1e-6), tgt,
                                       reduction="none").flatten(1).sum(-1).mean()
        kl = _kl(mu[:, -1], logvar[:, -1])
        return recon + kl_weight * kl, {"recon": float(recon), "kl": float(kl)}


class ExtrinsicRoadEncoder(nn.Module):
    """Stage 2: physical head on a FROZEN stage-1 latent.

    Takes the per-frame posterior means, concatenates the window, and regresses the
    curvature profile. The head is the reference implementation's MLP
    ([256,128,64], dropout 0.1); `head="transformer"` gives the variant the paper
    describes instead, since the two disagree and the choice should be visible.
    """

    def __init__(self, vae, n_offsets=10, frame_stack=LANE_FRAME_STACK,
                 head="mlp", freeze=True):
        super().__init__()
        self.vae, self.frame_stack, self.head_kind = vae, frame_stack, head
        if freeze:
            for p in self.vae.parameters():
                p.requires_grad = False
        d = vae.latent_dim
        if head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(d * frame_stack, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, n_offsets))
        elif head == "transformer":
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=512,
                                               batch_first=True, dropout=0.1)
            self.tf = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(d, n_offsets)
        else:
            raise ValueError(head)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.vae.parameters()):
            self.vae.eval()                     # keep the frozen stage in eval mode
        return self

    def forward(self, x):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.vae.parameters())):
            mu, _ = self.vae.encode(x)          # (B, FS, d); posterior mean, as in the reference
        if self.head_kind == "mlp":
            return self.head(mu.flatten(1))
        return self.head(self.tf(mu).mean(1))


class IntrinsicRoadEncoder(nn.Module):
    """A single encoder straight from images to the interpretable state.

    Following the conference formulation, the latent is partitioned in a fixed way
    into z* = [z_p, z_v]: z_p IS the curvature profile and carries the
    interpretability loss, z_v holds whatever else reconstruction needs and carries
    the KL. Reconstruction runs off the concatenation, so the physical part has to
    stay informative about the image rather than drifting free.

        L = L_recon + lambda_interp * L_interp(z_p, target) + beta * KL(z_v)

    z_p is deterministic: it is a physical quantity being regressed, and sampling it
    would inject noise into the very output the dynamics consumes. Only z_v is
    stochastic, which is also how the conference objective applies its KL.
    """

    def __init__(self, n_offsets=10, frame_stack=LANE_FRAME_STACK, visual_dim=118,
                 frame_dim=128):
        super().__init__()
        self.n_offsets, self.frame_stack, self.visual_dim = n_offsets, frame_stack, visual_dim
        self.enc = _FrameConv(1)
        # Project each frame BEFORE concatenating. Flattening 15 x 2048 straight into
        # a linear layer costs ~15.7M parameters on its own and would put this variant
        # an order of magnitude above the others, which defeats the comparison the
        # conference version sets up ("comparable parameter count ... architectural
        # efficacy rather than model capacity").
        self.frame_proj = nn.Linear(self.enc.out_dim, frame_dim)
        self.aggregate = nn.Sequential(
            nn.Linear(frame_dim * frame_stack, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.fc_p = nn.Linear(256, n_offsets)                 # physical, deterministic
        self.fc_v_mu = nn.Linear(256, visual_dim)             # visual, stochastic
        self.fc_v_logvar = nn.Linear(256, visual_dim)
        self.dec = _FrameDeconv(n_offsets + visual_dim, 1)

    def encode(self, x):
        b, fs = x.shape[:2]
        h = self.enc(x.reshape(b * fs, 1, *x.shape[2:]))
        h = self.frame_proj(h).view(b, -1)
        h = self.aggregate(h)
        return self.fc_p(h), self.fc_v_mu(h), self.fc_v_logvar(h)

    def forward(self, x):
        """The interpretable output alone, so this drops into the same call sites
        as RoadContextEncoder at evaluation time."""
        return self.encode(x)[0]

    def loss(self, x, target, lambda_interp=1000.0, beta=1.0):
        """Reconstruct the LAST frame only.

        Here it is the better target as well as the cheaper one: the latent is a
        single window-level vector, and z_p is the curvature profile ahead of NOW.
        Demanding that one vector reconstruct all fifteen frames spends z_v's
        capacity on history the task does not use. See ExtrinsicVisionVAE.forward
        for the memory arithmetic that makes the alternative impractical.
        """
        z_p, mu_v, logvar_v = self.encode(x)
        z_v = mu_v + torch.randn_like(mu_v) * (0.5 * logvar_v).exp()
        rec = self.dec(torch.cat([z_p, z_v], -1)).squeeze(1)
        recon = F.binary_cross_entropy(rec.clamp(1e-6, 1 - 1e-6), x[:, -1],
                                       reduction="none").flatten(1).sum(-1).mean()
        interp = F.mse_loss(z_p, target)
        kl = _kl(mu_v, logvar_v)
        total = recon + lambda_interp * interp + beta * kl
        return total, {"recon": float(recon), "interp": float(interp), "kl": float(kl)}
