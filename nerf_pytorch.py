import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder:
    def __init__(self, input_dims=3, num_freqs=10, include_input=True):
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.**torch.linspace(0., num_freqs - 1, num_freqs)

    def embed(self, inputs):
        if self.freq_bands.device != inputs.device:
            self.freq_bands = self.freq_bands.to(inputs.device)

        embed_list = []
        if self.include_input:
            embed_list.append(inputs)
        
        for freq in self.freq_bands:
            embed_list.append(torch.sin(inputs * freq))
            embed_list.append(torch.cos(inputs * freq))
            
        return torch.cat(embed_list, dim=-1)

class FastNeRF(nn.Module):
    def __init__(self, D=4, W=128, input_ch=63, input_ch_views=27):
        super(FastNeRF, self).__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.views_linears = nn.Linear(input_ch_views + W, W // 2)
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [63, 27], dim=-1)
        h = input_pts
        for l in self.pts_linears:
            h = F.relu(l(h))
        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        h = F.relu(self.views_linears(h))
        rgb = self.rgb_linear(h)
        return torch.cat([rgb, alpha], -1)

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
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

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=c2w.device), torch.linspace(0, H-1, H, device=c2w.device), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
