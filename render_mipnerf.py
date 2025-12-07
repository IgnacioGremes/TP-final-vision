import torch
import numpy as np
import imageio
import tqdm
import os
from mipnerf_pytorch import MipNeRF, conical_frustum_to_gaussian, raw2outputs_mip
from load_llff import load_llff_data

CONFIG = {
    'model_path': './logs/fern_mipnerf_final_BEST.pth', 
    'datadir': './data/nerf_llff_data/fern',
    'factor': 4,
    'N_samples': 128,
    'chunk': 4096,
    'layers': 8,
    'neurons': 256

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rays_mip_render(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=c2w.device), 
                          torch.linspace(0, H-1, H, device=c2w.device), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    base_radius = (2 / np.sqrt(12)) / (2 * K[0,0])
    radii = torch.ones_like(rays_d[..., :1]) * base_radius
    return rays_o, rays_d, radii

def render_image_mip(model, rays_o, rays_d, radii, near, far):
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3); rays_d = rays_d.reshape(-1, 3); radii = radii.reshape(-1, 1)
    rgb_frames = []
    
    for i in range(0, rays_o.shape[0], CONFIG['chunk']):
        b_o = rays_o[i:i+CONFIG['chunk']]; b_d = rays_d[i:i+CONFIG['chunk']]; b_r = radii[i:i+CONFIG['chunk']]
        
        t_vals = torch.linspace(near, far, CONFIG['N_samples'] + 1, device=device).expand(b_o.shape[0], CONFIG['N_samples'] + 1)
        t0 = t_vals[..., :-1]; t1 = t_vals[..., 1:]
        
        means, covs = conical_frustum_to_gaussian(b_o[:,None,:], b_d[:,None,:], t0, t1, b_r)
        viewdirs = b_d / torch.norm(b_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(means.shape)
        
        with torch.no_grad():
            raw = model(means, covs, viewdirs)
            rgb, _, _ = raw2outputs_mip(raw, (t0+t1)/2, b_d)
        rgb_frames.append(rgb.cpu().numpy())
        
    return np.concatenate(rgb_frames, 0).reshape(H, W, 3)

def run_render():
    print("--- RENDERIZANDO MIP-NERF ---")
    _, poses, bds, render_poses, _ = load_llff_data(CONFIG['datadir'], CONFIG['factor'], recenter=True, bd_factor=.75)
    
    H, W = int(poses[0,0,-1]/CONFIG['factor']), int(poses[0,1,-1]/CONFIG['factor'])
    focal = (poses[0,2,-1]/CONFIG['factor'])
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    near, far = bds.min() * 0.9, bds.max() * 1.0

    model = MipNeRF(D=CONFIG['layers'], W=CONFIG['neurons']).to(device)
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    else: return

    model.eval()
    frames = []
    for c2w in tqdm.tqdm(render_poses):
        c2w = torch.Tensor(c2w[:3, :4]).to(device)
        rays_o, rays_d, radii = get_rays_mip_render(H, W, K, c2w)
        rgb = render_image_mip(model, rays_o, rays_d, radii, near, far)
        frames.append((np.clip(rgb, 0, 1)*255).astype(np.uint8))

    imageio.mimwrite('./logs/video_mipnerf.mp4', frames, fps=30, quality=8)
    print("Video guardado.")

if __name__ == '__main__':
    run_render()
