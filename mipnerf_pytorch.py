import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IntegratedEmbedder:
    def __init__(self, input_dims=3, num_freqs=10):
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.freq_bands = 2.**torch.linspace(0., num_freqs - 1, num_freqs)

    def embed(self, means, covs):
        if self.freq_bands.device != means.device:
            self.freq_bands = self.freq_bands.to(means.device)

        scaled_means = means[..., None] * self.freq_bands
        scaled_covs = covs[..., None] * (self.freq_bands**2)

        sines = torch.sin(scaled_means) * torch.exp(-0.5 * scaled_covs)
        cosines = torch.cos(scaled_means) * torch.exp(-0.5 * scaled_covs)

        return torch.cat([sines, cosines], dim=-1).reshape(means.shape[:-1] + (-1,))

class MipNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, num_freqs=10):
        super(MipNeRF, self).__init__()
        self.D = D
        self.W = W
        
        self.embed_pts = IntegratedEmbedder(input_dims=input_ch, num_freqs=num_freqs)
        self.embed_views = IntegratedEmbedder(input_dims=input_ch_views, num_freqs=4)
        
        in_ch_pts = input_ch * num_freqs * 2
        in_ch_views = input_ch_views * 4 * 2
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(in_ch_pts, W)] + 
            [nn.Linear(W, W) if i != 4 else nn.Linear(W + in_ch_pts, W) for i in range(D-1)]
        )
        
        self.views_linears = nn.Linear(in_ch_views + W, W // 2)
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, means, covs, viewdirs):
        embedded_pts = self.embed_pts.embed(means, covs)
        
        embedded_views = self.embed_views.embed(viewdirs, torch.zeros_like(viewdirs))
        
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i == 4:
                h = torch.cat([embedded_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, embedded_views], -1)
        h = self.views_linears(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        return torch.cat([rgb, alpha], -1)

def conical_frustum_to_gaussian(rays_o, rays_d, t0, t1, radii, diag=True):
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    
    means = rays_o + rays_d * mu[..., None]
    
    d_mag_sq = torch.sum(rays_d**2, dim=-1, keepdim=True)
    
    var_t = (hw**2) / 3 
    var_r = (radii**2 * (mu**2) / 4 + (radii**2 * hw**2) / 5 )
    
    covs = rays_d**2 * var_t[...,None] + (1 - rays_d**2 / (d_mag_sq + 1e-10)) * var_r[...,None]
    
    return means, covs

def raw2outputs_mip(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape, device=raw.device) * raw_noise_std
        
    sigma = F.relu(raw[...,3] + noise)
    alpha = 1.0 - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, depth_map, acc_map
