"""npe_1d.py
============

1D Neural Posterior Estimator (NPE) for parameter inference from gap
sequences of 1D quasicrystal projections.

Architecture
------------
    spacings (B, seq_len)
      -> Embedding_NPE                 -> (B, context_dim)
      -> [optional] concat log_mean_gap and Linear(+1 -> context_dim)
      -> ResNet-conditioned NSF chain  -> theta

Input scale
-----------
When a scalar log_mean_gap is provided at every call, it is
concatenated to the embedding vector and re-projected to context_dim
via ``self.scale_proj``. This gives the flow access to the absolute
scale of the input sequence (mean-nearest-neighbour distance), which
is otherwise removed by the ``mean_nn`` normalisation the dataset
applies before batching.

Notes for the supervisor
------------------------
* The previous ``NPE_model.py`` contained a double-assignment bug:
  the embedding was chosen via a ``config["embedding_type"]`` branch
  and then unconditionally re-assigned to ``Embedding_NPE`` on the
  following line. All existing 1D checkpoints were therefore trained
  with ``Embedding_NPE``, and this class uses ``Embedding_NPE``
  unconditionally to match. The ``embedding_type`` field in the
  config is ignored; if you want ``EmbeddingResNet``, edit this file
  directly and retrain.

* State-dict compatibility: submodule names ``embedding``,
  ``scale_proj``, ``flow`` and their internals are preserved so
  ``NPE_model.py`` checkpoints load without any key remapping.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from models.embedding_1d import Embedding_NPE
from models.nsf import Flow, TransformRQS, ResNetConditioner


class NPEModel(nn.Module):
    """Conditional NSF for a 1D observation sequence.

    Constructor takes a config dict (matches the original public
    interface). ``default_config`` returns a sensible starting point.
    """

    def __init__(self, config: dict):
        super().__init__()

        context_dim = config["context_dim"]

        # Note: the historical NPE_model.py silently forced Embedding_NPE
        # regardless of ``embedding_type`` due to a double-assignment
        # bug. Preserved here so existing checkpoints load.
        self.embedding = Embedding_NPE(
            seq_len=config["seq_len"],
            kernel_size=config.get("kernel_size", 9),
            stride=config.get("stride", 1),
            padding=config.get("padding", 4),
            dropout_conv=config.get("dropout_conv", 0.1),
            pooling_ratio=config.get("pooling_ratio", 4),
            d_out=context_dim,
        )

        # Scalar log_mean_gap conditioning: cat (embedding, log_mean_gap)
        # then re-project to context_dim.
        self.scale_proj = nn.Linear(context_dim + 1, context_dim)

        theta_dim = config["theta_dim"]
        K = config["K"]
        B = config["B"]
        n_steps = config["n_flow_steps"]
        hidden_dim = config["hidden_dim_conditioner"]
        num_blocks = config["num_conditioner_blocks"]

        transforms = []
        for i in range(n_steps):
            mask = torch.zeros(theta_dim, dtype=torch.bool)
            mask[i % theta_dim] = True
            transforms.append(
                TransformRQS(
                    dAB=theta_dim,
                    K=K,
                    B=B,
                    dC=context_dim,
                    mask=mask,
                    conditioning_NN=ResNetConditioner,
                    hidden_dim_conditioner=hidden_dim,
                    min_height=1e-3,
                    min_derivative=1e-3,
                    min_width=1e-3,
                    num_blocks=num_blocks,
                )
            )

        self.flow = Flow(transforms=transforms, dim=theta_dim)

    # ---- context construction ------------------------------------------

    def _context(self, spacings, log_mean_gap=None):
        """Embed the observation, optionally enriching with scale info."""
        context = self.embedding(spacings)                     # (B, dC)
        if log_mean_gap is not None:
            if log_mean_gap.dim() == 1:
                log_mean_gap = log_mean_gap.unsqueeze(-1)      # (B, 1)
            context = torch.cat([context, log_mean_gap], dim=-1)
            context = self.scale_proj(context)                 # (B, dC)
        return context

    # ---- training / inference API --------------------------------------

    def compute_loss(self, spacings, theta, log_mean_gap=None):
        """Negative log-probability loss (scalar) for training.

        Args
        ----
        spacings     : (B, seq_len)
        theta        : (B, theta_dim)
        log_mean_gap : (B,) or (B, 1), optional
        """
        context = self._context(spacings, log_mean_gap)
        z, log_det = self.flow.inverse(theta, context)
        log_prob_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob_z + log_det
        return -log_prob.mean()

    def sample(self, spacings, n_samples: int, log_mean_gap=None):
        """Draw posterior samples p(theta | spacings).

        Args
        ----
        spacings     : (1, seq_len) single observation.
        n_samples    : int, number of samples.
        log_mean_gap : (1,) scalar, optional.

        Returns
        -------
        (n_samples, theta_dim)
        """
        context = self._context(spacings, log_mean_gap)
        context = context.expand(n_samples, -1)
        z = self.flow.base.sample((n_samples,)).to(context.device)
        theta_samples, _ = self.flow.forward(z, context)
        return theta_samples

    def log_prob(self, spacings, theta, log_mean_gap=None):
        """log p(theta | spacings) per element."""
        context = self._context(spacings, log_mean_gap)
        z, log_det = self.flow.inverse(theta, context)
        log_prob_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        return log_prob_z + log_det


def default_config() -> dict:
    """Matches what was used in the thesis."""
    return {
        "seq_len": 2048,
        "context_dim": 128,
        "theta_dim": 2,
        "K": 8,
        "B": 3,
        "n_flow_steps": 6,
        "hidden_dim_conditioner": 128,
        "num_conditioner_blocks": 2,
        "kernel_size": 9,
        "padding": 4,
        "stride": 1,
        "pooling_ratio": 4,
        "dropout_conv": 0.1,
        "embedding_type": "cnn",
    }