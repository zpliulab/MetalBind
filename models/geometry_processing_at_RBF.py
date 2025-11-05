import math
import numpy as np
import os
# os.environ['CUDA_PATH']='/home/aoli/tools/cuda10.0'
import torch
from pykeops.torch import LazyTensor
import torch.nn as nn
from math import sqrt

def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices

def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x

def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

          normals        ->             uv
    (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

    Args:
        normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns:
        (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv

#  Fast tangent convolution layer ===============================================
class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    N.B.: This workaround fixes a bug that will be fixed in ulterior KeOp releases. 
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()

class dMaSIFConv(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, radius=6.0, hidden_units=None, cheap=False,
        num_rbf=8, rbf_max=2.0, rbf_proj_dim=4
    ):
        """
        RBF  dMaSIFConv。
            num_rbf: RBFs 8
            rbf_max: centers 2.0
            rbf_proj_dim: 4
        """
        super(dMaSIFConv, self).__init__()
        self.Input = in_channels
        self.Output = out_channels
        self.Radius = radius
        self.Hidden = self.Output if hidden_units is None else hidden_units
        self.Cuts = 8
        self.cheap = cheap

        # RBF params
        self.num_rbf = int(num_rbf)
        centers = torch.linspace(0.0, float(rbf_max), self.num_rbf)
        self.register_buffer("rbf_centers", centers)

        spacing = rbf_max / (num_rbf - 1)
        sigma = 0.6 * spacing

        # learnable log sigma
        self.rbf_log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))  #  sigma = 0.2

        # projection dimension for RBF -> reduce computation
        self.rbf_proj_dim = int(rbf_proj_dim)
        # small linear projection: num_rbf -> rbf_proj_dim
        self.rbf_proj = nn.Linear(self.num_rbf, self.rbf_proj_dim, bias=True)

        self.res_proj = nn.Linear(self.Input, self.Output)

        # Multi-head setup
        self.heads_dim = min(8, self.Hidden)
        if self.Hidden % self.heads_dim != 0:
            raise ValueError("Hidden must be multiple of heads_dim")
        self.n_heads = self.Hidden // self.heads_dim

        # Input and output MLPs
        self.net_in = nn.Sequential(
            nn.Linear(self.Input, self.Hidden), nn.LeakyReLU(0.2),
            nn.Linear(self.Hidden, self.Hidden), nn.LeakyReLU(0.2)
        )
        self.norm_in = nn.GroupNorm(num_groups=4, num_channels=self.Hidden)

        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output), nn.LeakyReLU(0.2),
            nn.Linear(self.Output, self.Output), nn.LeakyReLU(0.2)
        )
        self.norm_out = nn.GroupNorm(num_groups=4, num_channels=self.Output)

        # Attention projections
        self.W_q = nn.Linear(self.Hidden, self.Hidden)
        self.W_k = nn.Linear(self.Hidden, self.Hidden)
        self.W_v = nn.Linear(self.Hidden, self.Hidden)

        # Spatial filter MLP: input dim = 4 (local coords + n_dot) + rbf_proj_dim (投影后)
        conv_in_dim = 4 + self.rbf_proj_dim
        if cheap:
            self.conv = nn.Sequential(nn.Linear(conv_in_dim, self.Hidden), nn.ReLU())
        else:
            self.conv = nn.Sequential(
                nn.Linear(conv_in_dim, self.Cuts), nn.ReLU(),
                nn.Linear(self.Cuts, self.Hidden)
            )

        # Initialize conv & rbf_proj weights
        with torch.no_grad():
            # conv[0]
            nn.init.normal_(self.conv[0].weight)
            nn.init.uniform_(self.conv[0].bias)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()
            if not cheap:
                nn.init.uniform_(self.conv[2].weight, -1/np.sqrt(self.Cuts), 1/np.sqrt(self.Cuts))
                nn.init.normal_(self.conv[2].bias)
                self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()

            # rbf_proj small init
            nn.init.xavier_uniform_(self.rbf_proj.weight)
            nn.init.zeros_(self.rbf_proj.bias)

    def forward(self, points, nuv, features, ranges=None):
        # ensure contiguity
        points = points.contiguous()
        nuv = nuv.contiguous()
        features = features.contiguous()

        # 1. Input transform
        h = self.net_in(features)  # (N, H)
        h = h.contiguous()
        h = h.transpose(1, 0).unsqueeze(0).contiguous()  # (1, H, N)
        h = self.norm_in(h)
        h = h.squeeze(0).transpose(1, 0).contiguous()  # (N, H)

        # Attention projections
        Q = self.W_q(h).contiguous()  # (N, H)
        K = self.W_k(h).contiguous()  # (N, H)
        V = self.W_v(h).contiguous()  # (N, H)

        # Normalize coords by Radius (same as before)
        pts = (points / (math.sqrt(2.0) * self.Radius)).contiguous()
        x_i = LazyTensor(pts[:, None, :])  # (N,1,3)
        x_j = LazyTensor(pts[None, :, :])  # (1,N,3) -> broadcast to (N,N,3)

        normals = nuv[:, 0, :].detach().contiguous()
        nuv_i = LazyTensor(nuv.view(-1, 1, 9).contiguous())
        n_i = nuv_i[:, :, :3]
        n_j = LazyTensor(normals[None, :, :])

        head_out = []
        d_k = self.heads_dim

        # Pre-convert rbf_proj weights to LazyTensor representations for KeOps matvecmult
        # rbf_proj.weight shape: (rbf_proj_dim, num_rbf)
        a_rbf = LazyTensor(self.rbf_proj.weight.contiguous().view(1, 1, -1))  # for matvecmult
        b_rbf = LazyTensor(self.rbf_proj.bias.contiguous().view(1, 1, -1))

        for h_id in range(self.n_heads):
            start, end = h_id * d_k, (h_id + 1) * d_k
            q = LazyTensor(Q[:, None, start:end].contiguous())   # (N,1,d_k)
            k = LazyTensor(K[None, :, start:end].contiguous())   # (1,N,d_k)
            v = LazyTensor(V[None, :, start:end].contiguous())   # (1,N,d_k)

            # Pseudo-geodesic distance -> W_ij (保留原实现)
            d2_geo = ((x_j - x_i)**2).sum(-1) * ((2 - (n_i | n_j))**2)
            alpha = 3.0
            W_ij = (1.0 + d2_geo / alpha) ** (-alpha)

            # Use geo-based distance for RBF as well to ensure geometric consistency
            base_d = d2_geo.sqrt()  # (N,N,1) LazyTensor

            # Build RBF features per-center
            sigma = (self.rbf_log_sigma.exp().clamp_min(1e-6))
            sigma_lt = LazyTensor(sigma.view(1, 1, 1))

            R = None
            for idx in range(self.num_rbf):
                ck = self.rbf_centers[idx].view(1, 1, 1)  # scalar
                ck_lt = LazyTensor(ck)
                rk = (-((base_d - ck_lt) ** 2) / (2.0 * (sigma_lt ** 2))).exp()  # (N,N,1)
                R = rk if R is None else R.concat(rk)  # (N,N,K) at end

            # Project RBF K -> rbf_proj_dim using LazyTensor matvecmult (keeps laziness)
            # a_rbf.matvecmult(R) produces (N,N,rbf_proj_dim)
            R_proj = (a_rbf.matvecmult(R) + b_rbf)  # (N, N, rbf_proj_dim)
            # optional non-linearity could be applied, but we'll keep linear mapping for lightness
            # R_proj = R_proj.relu()

            # Local coords + n_i|n_j
            X = nuv_i.matvecmult(x_j - x_i)   # (N, N, 3)
            X = X.concat((n_i | n_j))         # (N, N, 4)

            # Concat projected RBF -> (N, N, 4 + rbf_proj_dim)
            X = X.concat(R_proj)

            # conv (cheap / normal)
            if self.cheap:
                AB = torch.cat((
                    self.conv[0].weight[start:end].contiguous(),         # (d_k, conv_in_dim)
                    self.conv[0].bias[start:end].contiguous().view(-1, 1)  # (d_k, 1)
                ), dim=1)  # (d_k, conv_in_dim+1)
                AB_lt = LazyTensor(AB.view(1, 1, -1))
                F = AB_lt.matvecmult(X.concat(LazyTensor(1))).relu()
            else:
                a1 = LazyTensor(self.conv[0].weight.contiguous().view(1, 1, -1))
                b1 = LazyTensor(self.conv[0].bias.contiguous().view(1, 1, -1))
                a2 = LazyTensor(self.conv[2].weight[start:end].contiguous().view(1, 1, -1))
                b2 = LazyTensor(self.conv[2].bias[start:end].contiguous().view(1, 1, -1))
                F = (a1.matvecmult(X) + b1).relu()
                F = (a2.matvecmult(F) + b2).relu()

            # Attention scores
            S_ij = (q * k).sum(-1) / math.sqrt(d_k)

            # Combine window, filter, value
            V_ij = W_ij * F * v

            # Aggregate with KeOps softmax-weighted sum
            out_head = S_ij.sumsoftmaxweight(V_ij, dim=1)

            # Ensure contiguous / convert for concatenation downstream
            head_out.append(ContiguousBackward.apply(out_head))

        agg = torch.cat(head_out, dim=1)  # (N, Hidden)

        # Output transform + GroupNorm + residual
        out = self.net_out(agg).contiguous()
        out = out.transpose(1, 0).unsqueeze(0).contiguous()
        out = self.norm_out(out)
        out = out.squeeze(0).transpose(1, 0).contiguous()
        out = out + self.res_proj(features)

        return out