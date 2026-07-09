"""
Implementation of the neural spline flow according to "Neural Spline Flow, durkan et al. 2019"
Current implementation supports 1D input data, but this can be expanded easily by adjusting the context network. 
As long as the output of the context network is of the form (...,d*(3k-1)), 
with k the number of knots and d the number of dimensions to transform,
the subsequent normalising flow will work. 
The transform is currently only setup to use a standard normal distribution as the latent distribution. 

usage:

    from nsf_posterior import NSFPosterior

    model = NSFPosterior(
        seq_len=2048,
        theta_dim=2, 
        context_dim=128, 
        n_flow_steps=6, 
        n_spline_knots=8, 
        spline_bound=3.0, 
    )

    if using a custom context (this functions also with data in higher dimension than 1)"
    # Training: compute loss on (observation, parameter) pairs
    loss = model.compute_loss(observations, theta)

    # Inference: sample posterior given a single observation
    samples = model.sample(observation, n_samples=5000)

    # Evaluation: log p(theta | observation)
    log_p = model.log_prob(observations, theta)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# Activation functions that map conditioning network -> spline parameters

def _activate_positions(params, bound, n_knots, min_width, min_height):
    """converts an context network output to raw positions. Requires (..., 3K-1) raw parameters from a conditioning network.
       
    Arguments:
        params:     (..., 3K-1) raw parameters from conditioner.
        bound:      Spline boundary [-bound, bound].
        n_knots:    Number of spline bins (K).
        min_width:  Minimum bin width.
        min_height: Minimum bin height.

    Returns:
        widths:  (..., K+1) x-coordinates of knots.
        heights: (..., K+1) y-coordinates of knots.
    """
    w_raw = params[..., :n_knots]
    w_norm = torch.softmax(w_raw, dim=-1) * (1 - n_knots * min_width) + min_width
    cum_w = F.pad(torch.cumsum(w_norm, dim=-1), (1, 0), value=0.0)
    widths = 2 * bound * cum_w - bound
    widths[..., 0], widths[..., -1] = -bound, bound

    h_raw = params[..., n_knots : 2 * n_knots]
    h_norm = torch.softmax(h_raw, dim=-1) * (1 - n_knots * min_height) + min_height
    cum_h = F.pad(torch.cumsum(h_norm, dim=-1), (1, 0), value=0.0)
    heights = 2 * bound * cum_h - bound
    heights[..., 0], heights[..., -1] = -bound, bound

    return widths, heights


def _activate_derivatives(params, n_knots, min_derivative=1e-3,
                          identity_init=True):
    """Convert raw outputs to positive knot derivatives.

    When ``identity_init=True`` the softplus bias is chosen so that
    zero-initialised conditioner outputs produce derivatives ≈ 1,
    making the initial flow close to the identity transform.
    """
    raw = params[..., 2 * n_knots :]
    if identity_init:
        beta = torch.log(raw.new_tensor(2.0)) / (1.0 - min_derivative)
        c = (1.0 / beta) * torch.log(
            torch.exp(beta * (1.0 - min_derivative)) - 1.0
        )
    else:
        beta = raw.new_tensor(1.0)
        c = torch.log(torch.exp(raw.new_tensor(1.0 - min_derivative)) - 1.0)

    padded = F.pad(raw, (1, 1), value=float(c))
    return min_derivative + F.softplus(padded, beta=float(beta))

# The forward and inverse definitions of the RQS + implementation into the full transform

def _spline_forward(x, widths, heights, derivatives, idx, eps=1e-6):
    """Evaluate spline x -> y and compute log |dy/dx|."""
    x_lo = torch.gather(widths, 2, idx - 1).squeeze(-1)
    x_hi = torch.gather(widths, 2, idx).squeeze(-1)
    y_lo = torch.gather(heights, 2, idx - 1).squeeze(-1)
    y_hi = torch.gather(heights, 2, idx).squeeze(-1)
    d_lo = torch.gather(derivatives, 2, idx - 1).squeeze(-1)
    d_hi = torch.gather(derivatives, 2, idx).squeeze(-1)

    dx = (x_hi - x_lo).clamp(min=eps)
    sk = (y_hi - y_lo) / dx
    eta = (x - x_lo) / dx

    num = (y_hi - y_lo) * (sk * eta ** 2 + d_lo * eta * (1 - eta))
    den = sk + (d_hi + d_lo - 2 * sk) * eta * (1 - eta)
    y = y_lo + num / den.clamp(min=eps)

    num_dy = sk ** 2 * (
        d_hi * eta ** 2 + 2 * sk * eta * (1 - eta) + d_lo * (1 - eta) ** 2
    )
    den_dy = (sk + (d_hi + d_lo - 2 * sk) * eta * (1 - eta)) ** 2
    log_det = torch.log((num_dy / den_dy).clamp(min=eps))
    return y, log_det


def _spline_inverse(y, widths, heights, derivatives, idx, eps=1e-6):
    """Evaluate spline inverse y -> x and compute log |dx/dy|."""
    x_lo = torch.gather(widths, 2, idx - 1).squeeze(-1)
    x_hi = torch.gather(widths, 2, idx).squeeze(-1)
    y_lo = torch.gather(heights, 2, idx - 1).squeeze(-1)
    y_hi = torch.gather(heights, 2, idx).squeeze(-1)
    d_lo = torch.gather(derivatives, 2, idx - 1).squeeze(-1)
    d_hi = torch.gather(derivatives, 2, idx).squeeze(-1)

    dx = (x_hi - x_lo).clamp(min=eps)
    sk = (y_hi - y_lo) / dx

    a = (y_hi - y_lo) * (sk - d_lo) + (y - y_lo) * (d_hi + d_lo - 2 * sk)
    b = (y_hi - y_lo) * d_lo - (y - y_lo) * (d_hi + d_lo - 2 * sk)
    c = -sk * (y - y_lo)

    discriminant = b ** 2 - 4 * a * c
    eta = (2 * c) / (-b - torch.sqrt(discriminant.clamp(min=0)))
    eta = eta.clamp(0.0, 1.0)

    x = eta * dx + x_lo

    # Reuse forward derivative formula for log |dy/dx|, then negate
    num_dy = sk ** 2 * (
        d_hi * eta ** 2 + 2 * sk * eta * (1 - eta) + d_lo * (1 - eta) ** 2
    )
    den_dy = (sk + (d_hi + d_lo - 2 * sk) * eta * (1 - eta)) ** 2
    log_det = -torch.log((num_dy / den_dy).clamp(min=eps))
    return x, log_det


def rational_quadratic_spline(x, params, bound, n_knots, inverse=False,
                              min_width=1e-3, min_height=1e-3,
                              min_derivative=1e-3, eps=1e-6):
    
    """Applies a rational quadratic spline to input x. 1 pass-through

    Points outside of boundaries are transformed by unity. 

    arguments:
        x:       (batch, dim) input values.
        params:  (batch, dim, 3K-1) raw spline parameters.
        bound:   Scalar, spline boundary.
        n_knots: Number of spline bins (K).
        inverse: If True, compute the inverse transform.

    returns:
        y:       (batch, dim) transformed values.
        log_det: (batch, dim) log |dy/dx| per element.
    """

    widths, heights = _activate_positions(params, bound, n_knots,
                                          min_width, min_height)
    derivatives = _activate_derivatives(params, n_knots, min_derivative,
                                        identity_init=True)

    inside = (x >= -bound) & (x <= bound)
    x_clamped = x.clamp(-bound + eps, bound - eps)

    if inverse:
        search_vals = heights.clone()
        search_vals[..., -1] += eps
        idx = torch.searchsorted(search_vals, x_clamped.unsqueeze(-1)).clamp(1, n_knots)
        y_in, ld_in = _spline_inverse(x_clamped, widths, heights, derivatives, idx, eps)
    else:
        search_vals = widths.clone()
        search_vals[..., -1] += eps
        idx = torch.searchsorted(search_vals, x_clamped.unsqueeze(-1)).clamp(1, n_knots)
        y_in, ld_in = _spline_forward(x_clamped, widths, heights, derivatives, idx, eps)

    zero = torch.zeros_like(ld_in)
    y = torch.where(inside, y_in, x)
    log_det = torch.where(inside, ld_in, zero)
    return y, log_det

def RQS(x, params, B, K, inverse=False,
        min_width=1e-3, min_height=1e-3,
        min_derivative=1e-3, eps=1e-6):
    """Alias with the kwarg names TransformRQS uses.

    ``rational_quadratic_spline`` takes (bound, n_knots); ``TransformRQS``
    was written against (B, K). Same function otherwise.
    """
    return rational_quadratic_spline(
        x, params,
        bound=B, n_knots=K,
        inverse=inverse,
        min_width=min_width,
        min_height=min_height,
        min_derivative=min_derivative,
        eps=eps,
    )
# Conditioning networks


class ResBlock(nn.Module):
    """Pre-activation residual block."""
    def __init__(self, dim, dropout=0.0, batch_norm=False):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.bn1(x)
        h = F.relu(h)
        h = self.linear1(h)
        h = self.dropout(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.linear2(h)
        h = self.dropout(h)
        return x + h


class Conditioner(nn.Module):
    """resnet MLP: maps (x_passthrough, context) -> spline parameters.

    Output layer is zero-initialised so the flow starts near identity.
    """

    def __init__(self, d_in, d_out, hidden_dim, d_context=0,
                 num_blocks=2, dropout=0.0):
        super().__init__()
        self.d_context = d_context
        self.proj = nn.Linear(d_in + d_context, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(hidden_dim, d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_pass, context=None):
        if context is not None and self.d_context > 0:
            x_pass = torch.cat([x_pass, context], dim=-1)
        h = self.proj(x_pass)
        h = self.blocks(h)
        return self.out(h)


# Coupling layer


class CouplingRQS(nn.Module):
    """Single coupling layer: split x by mask, transform one half with an
    RQS whose parameters are conditioned on the other half + context."""

    def __init__(self, dim, mask, n_knots, bound, context_dim=0,
                 hidden_dim=128, num_blocks=2, dropout=0.0,
                 min_width=1e-3, min_height=1e-3, min_derivative=1e-3):
        super().__init__()
        mask = mask.bool()
        self.register_buffer("mask", mask)
        self.n_knots = n_knots
        self.bound = bound
        self.min_width = min_width
        self.min_height = min_height
        self.min_derivative = min_derivative

        d_pass = int(mask.sum().item())
        d_transform = int((~mask).sum().item())
        self.d_transform = d_transform

        self.conditioner = Conditioner(
            d_in=d_pass,
            d_out=d_transform * (3 * n_knots - 1),
            hidden_dim=hidden_dim,
            d_context=context_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

    def forward(self, x, context=None, inverse=False):
        x_pass = x[:, self.mask]
        x_transform = x[:, ~self.mask]

        params = self.conditioner(x_pass, context)
        params = params.view(x.size(0), self.d_transform, 3 * self.n_knots - 1)

        y_transform, log_det = rational_quadratic_spline(
            x_transform, params,
            bound=self.bound, n_knots=self.n_knots, inverse=inverse,
            min_width=self.min_width, min_height=self.min_height,
            min_derivative=self.min_derivative,
        )

        y = torch.empty_like(x)
        y[:, self.mask] = x_pass
        y[:, ~self.mask] = y_transform
        return y, log_det


# Flow (chain of coupling layers + base distribution)

class Flow(nn.Module):
    """Chain of coupling transforms with a standard normal base."""

    def __init__(self, transforms, dim):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.dim = dim
        self.base = Independent(
            Normal(torch.zeros(dim), torch.ones(dim)),
            reinterpreted_batch_ndims=1,
        )

    def forward(self, z, context=None):
        """z -> x (sampling direction)."""
        log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        x = z
        for t in self.transforms:
            x, ld = t(x, context, inverse=False)
            log_det += ld.view(z.size(0), -1).sum(-1)
        return x, log_det

    def inverse(self, x, context=None):
        """x -> z (training direction)."""
        log_det = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        z = x
        for t in reversed(self.transforms):
            z, ld = t(z, context, inverse=True)
            log_det += ld.view(x.size(0), -1).sum(-1)
        return z, log_det

class TransformRQS(nn.Module):
    def __init__(self, 
                 dAB, 
                 K, 
                 B, 
                 dC, 
                 mask, 
                 conditioning_NN, 
                 hidden_dim_conditioner,
                 min_height,
                 min_derivative,
                 min_width,
                 num_blocks):
        super().__init__()

        self.dAB = dAB
        self.K = K
        self.B = B
        self.min_height = min_height
        self.min_width = min_width
        self.min_derivative = min_derivative
        self.num_blocks = num_blocks
        self.dC = dC if dC != 0 else None

        mask = mask.bool()
        self.register_buffer("mask", mask)
        dA, dB = int(mask.sum().item()), int((~mask).sum().item())
        self.dA = dA
        self.dB = dB

        self.condition = conditioning_NN(d_in=dA,
                                        d_out=dB*(3*K-1),
                                        channels=hidden_dim_conditioner,
                                        d_context = self.dC ,
                                        activation=F.relu,
                                        num_blocks=num_blocks,
                                        dropout=0.0,
                                        batch_norm=False)

    def _split(self, x):
        return x[:, self.mask], x[:, ~self.mask]
    
    def _merge(self, xA, xB):
        y = torch.empty(xA.size(0), self.dA + self.dB, device=xA.device, dtype=xA.dtype)
        y[:, self.mask] = xA
        y[:, ~self.mask] = xB
        return y
    
    def forward(self, x, context, inverse):

        xA, xB = self._split(x)
        params = self.condition(xA, context)
        params = params.view(x.size(0), self.dB, 3*self.K - 1)
        
        yB, ld = RQS(xB, 
                     params=params, 
                     B=self.B, 
                     K=self.K, 
                     inverse=inverse,  
                     min_derivative=self.min_derivative, min_height=self.min_height, min_width=self.min_width,)
        
        y = self._merge(xA, yB)

        return y, ld

# 1D CNN context for observations


class Context(nn.Module):
    """Simple 1D CNN that maps (batch, seq_len) -> (batch, d_out).

    Appends global mean and log-std statistics before the output head,
    giving the network access to scale information that pooling discards.
    """

    def __init__(self, seq_len, d_out=128, kernel_size=9, padding=4,
                 dropout=0.1, pooling_ratio=4):
        super().__init__()
        channels = [1, 16, 32, 32]
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=padding),
                nn.BatchNorm1d(c_out),
                nn.SiLU(),
                nn.AvgPool1d(pooling_ratio, pooling_ratio),
                nn.Dropout1d(dropout),
            ]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1] + 2, d_out)  # +2 for mean, log-std

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        mean = x.mean(dim=-1)                          # (batch, 1)
        log_std = x.std(dim=-1).clamp(min=1e-8).log()  # (batch, 1)
        stats = torch.cat([mean, log_std], dim=1)       # (batch, 2)
        h = self.conv(x).mean(dim=-1)                   # (batch, C)
        return self.head(torch.cat([h, stats], dim=1))


class ContextResNet(nn.Module):
    """ResNet-style 1D CNN: (batch, seq_len) -> (batch, d_out).

    Deeper alternative to ``_Context`` for longer sequences.
    """

    def __init__(self, seq_len, d_out=128, kernel_size=9, padding=4,
                 dropout=0.1, pooling_ratio=4, n_res_blocks=4):
        super().__init__()
        ch = 128
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size, padding=padding),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[self._make_block(ch, kernel_size, dropout)
              for _ in range(n_res_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(pooling_ratio)
        self.fc = nn.Linear(ch * pooling_ratio, d_out)

    @staticmethod
    def _make_block(ch, kernel_size, dropout):
        pad = kernel_size // 2
        block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=pad),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel_size, padding=pad),
            nn.BatchNorm1d(ch),
        )
        # Wrap in a residual connection via a small helper
        class Res(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
            def forward(self, x):
                return F.relu(x + self.block(x))
        return Res(block)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))



# NSFPosterior


class NSFPosterior(nn.Module):
    
    """Conditional Neural Spline Flow for posterior estimation.

    Embeds a 1D observation sequence into a context vector (custom context networks allow for higher dimensional data), then uses
    a chain of Rational Quadratic Spline coupling layers to model p(theta | observation).

    arguments:
        seq_len:          Length of the input observation sequence.
        theta_dim:        Dimensionality of the parameter space.
        context_dim:      Size of the context vector.
        n_flow_steps:     Number of coupling layers.
        n_spline_knots:   Number of RQS bins (K).
        spline_bound:     Spline boundary (B); identity outside [-B, B].
        hidden_dim:       Hidden dimension of the conditioner MLPs.
        num_cond_blocks:  Residual blocks per conditioner.
        context:        ``"cnn"`` (light) or ``"resnet"`` (deeper).
        kernel_size:      Conv kernel size in the context.
        padding:          Conv padding in the context.
        dropout:          Dropout rate (context + conditioner).
        pooling_ratio:    Spatial pooling factor in the context.
        use_log_mean_gap: If True, ``compute_loss`` / ``sample`` /
                          ``log_prob`` accept an extra ``log_mean_gap``
                          scalar that is concatenated to the context.
        custom_context:   Custom context network. Output dimension has to match context_dim parameter. 
                          Custom_Context should be initialised outside of this model before being given as an argument.
    """

    def __init__(
        self,
        seq_len: int = 2048,
        theta_dim: int = 2,
        context_dim: int = 128,
        n_flow_steps: int = 6,
        n_spline_knots: int = 8,
        spline_bound: float = 3.0,
        hidden_dim: int = 128,
        num_cond_blocks: int = 2,
        context: str = "cnn",
        kernel_size: int = 9,
        padding: int = 4,
        dropout: float = 0.1,
        pooling_ratio: int = 4,
        use_log_mean_gap: bool = False,
        custom_context: bool = None
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.use_log_mean_gap = use_log_mean_gap

        # --- context ---
        emb_kwargs = dict(seq_len=seq_len, 
                          d_out=context_dim,
                          kernel_size=kernel_size, 
                          padding=padding,
                          dropout=dropout, 
                          pooling_ratio=pooling_ratio)
        
        if custom_context != None:
            self.context = custom_context
        elif context == "resnet":
            self.context = _ContextResNet(**emb_kwargs)
        elif context == "cnn":
            self.context = _Context(**emb_kwargs)
        

        # --- Optional scale projection (for log-mean-gap conditioning) ---
        if use_log_mean_gap:
            self.scale_proj = nn.Linear(context_dim + 1, context_dim)
        else:
            self.scale_proj = None

        # --- Coupling layers ---
        transforms = []
        for i in range(n_flow_steps):
            mask = torch.zeros(theta_dim, dtype=torch.bool)
            mask[i % theta_dim] = True
            transforms.append(
                CouplingRQS(
                    dim=theta_dim,
                    mask=mask,
                    n_knots=n_spline_knots,
                    bound=spline_bound,
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    num_blocks=num_cond_blocks,
                    dropout=0.0,
                )
            )
        self.flow = Flow(transforms, dim=theta_dim)

    # -- helpers --

    def _context(self, observations, log_mean_gap=None):
        ctx = self.context(observations)
        if self.scale_proj is not None and log_mean_gap is not None:
            if log_mean_gap.dim() == 1:
                log_mean_gap = log_mean_gap.unsqueeze(-1)
            ctx = self.scale_proj(torch.cat([ctx, log_mean_gap], dim=-1))
        return ctx

    @staticmethod
    def _log_prob_base(z):
        return -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)

    # -- public interface --

    def compute_loss(self, observations, theta, log_mean_gap=None):
        
        """Negative log-probability loss (scalar) for training.

        Args:
            observations: (batch, seq_len)
            theta:        (batch, theta_dim)
            log_mean_gap: (batch,) optional, requires ``use_log_mean_gap=True``.
        """

        ctx = self._context(observations, log_mean_gap)
        z, log_det = self.flow.inverse(theta, ctx)
        return -(self._log_prob_base(z) + log_det).mean()

    @torch.no_grad()
    def sample(self, observation, n_samples, log_mean_gap=None):
        
        """Draw posterior samples p(theta | observation).

        Args:
            observation: (1, seq_len) single observation.
            n_samples:   Number of samples to draw.
            log_mean_gap: (1,) optional scalar.

        Returns:
            (n_samples, theta_dim) tensor.
        """

        ctx = self._context(observation, log_mean_gap)
        ctx = ctx.expand(n_samples, -1)
        z = self.flow.base.sample((n_samples,)).to(ctx.device)
        samples, _ = self.flow.forward(z, ctx)
        return samples

    def log_prob(self, observations, theta, log_mean_gap=None):
        
        """Compute log p(theta | observations).

        Args:
            observations: (batch, seq_len)
            theta:        (batch, theta_dim)

        Returns:
            (batch,) log-probabilities.
        """

        ctx = self._context(observations, log_mean_gap)
        z, log_det = self.flow.inverse(theta, ctx)
        return self._log_prob_base(z) + log_det

class ResNetConditioner(nn.Module):
    """ResNet MLP conditioner for coupling layers.

    Concatenates passthrough variables xA with context vector,
    projects to hidden dim, passes through residual blocks,
    outputs spline parameters.

    Constructor and forward signatures match what TransformRQS expects.
    """

    def __init__(self, d_in, d_out, channels, d_context=None,
                 activation=None, num_blocks=2, dropout=0.0, batch_norm=False):
        super().__init__()
        input_dim = d_in + (d_context if d_context is not None else 0)
        self.d_context = d_context

        self.input_proj = nn.Linear(input_dim, channels)
        self.blocks = nn.ModuleList([
            ResBlock(channels, dropout, batch_norm)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(channels, d_out)

        # Zero-init output so flow starts near identity transform
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, xA, context=None):
        if context is not None and self.d_context is not None:
            x = torch.cat([xA, context], dim=-1)
        else:
            x = xA
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)