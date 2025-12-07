import torch
import numpy as np
import os
import tqdm
import json  
import matplotlib.pyplot as plt
from mipnerf_pytorch import MipNeRF, conical_frustum_to_gaussian, raw2outputs_mip
from load_llff import load_llff_data 

CONFIG = {
    'expname': 'fern_mipnerf_final',
    'datadir': './data/nerf_llff_data/fern',
    'factor': 4,          
    'N_samples': 128,     
    'N_iters': 100000,    
    'batch_size': 4096,
    'lrate': 5e-4,
    'i_val': 1000,
    'patience': 15,
    'layers': 8,
    'neurons': 256
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rays_mip(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=c2w.device), 
                          torch.linspace(0, H-1, H, device=c2w.device), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    
    base_radius = (2 / np.sqrt(12)) / (2 * K[0,0]) 
    radii = torch.ones_like(rays_d[..., :1]) * base_radius
    
    return rays_o, rays_d, radii

def render_full_image_mip(model, rays_o, rays_d, radii, near, far, chunk=CONFIG['batch_size']):
    """ Renderiza imagen completa usando la lógica de Conos de Mip-NeRF """
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3); rays_d = rays_d.reshape(-1, 3); radii = radii.reshape(-1, 1)
    rgb_frames = []
    
    for i in range(0, rays_o.shape[0], chunk):
        b_o = rays_o[i:i+chunk]; b_d = rays_d[i:i+chunk]; b_r = radii[i:i+chunk]
        
        t_vals = torch.linspace(near, far, CONFIG['N_samples'] + 1, device=device).expand(b_o.shape[0], CONFIG['N_samples'] + 1)
        t0 = t_vals[..., :-1]; t1 = t_vals[..., 1:]
        
        means, covs = conical_frustum_to_gaussian(b_o[:,None,:], b_d[:,None,:], t0, t1, b_r)
        viewdirs = b_d / torch.norm(b_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(means.shape)
        
        with torch.no_grad():
            raw = model(means, covs, viewdirs)
            rgb, _, _ = raw2outputs_mip(raw, (t0+t1)/2, b_d)
        
        rgb_frames.append(rgb)
        
    return torch.cat(rgb_frames, 0).reshape(H, W, 3)

def train():
    print(f"--- Mip-NeRF Training ({device}) ---")
    
    images, poses, bds, _, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'], recenter=True, bd_factor=.75)
    
    H, W = images.shape[1:3]
    original_H = poses[0, 0, -1]
    focal = poses[0, 2, -1]
    if H != original_H: focal *= (H / original_H)
    H, W = int(H), int(W)
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    bds = torch.Tensor(bds).to(device)

    if np.ndim(i_test) == 0: val_idx = int(i_test)
    else: val_idx = int(i_test[0])

    test_idx = 7 
    if test_idx == val_idx: test_idx = (val_idx + 1) % int(images.shape[0])

    all_indices = np.arange(int(images.shape[0]))
    i_train = np.array([i for i in all_indices if i not in [val_idx, test_idx]])
    
    print(f"Train: {len(i_train)} | Val: #{val_idx} | Test: #{test_idx}")

    model = MipNeRF(D=CONFIG['layers'], W=CONFIG['neurons']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lrate'])

    train_loss_history = []
    val_psnr_history = []
    iter_history = []
    best_psnr = 0.0
    patience_counter = 0
    os.makedirs('./logs', exist_ok=True)

    pbar = tqdm.tqdm(range(CONFIG['N_iters']))
    
    for i in pbar:
        img_idx = np.random.choice(i_train)
        target = images[img_idx]
        pose = poses[img_idx, :3, :4]
        
        rays_o, rays_d, radii = get_rays_mip(H, W, K, pose)
        
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H, device=device), torch.linspace(0, W-1, W, device=device), indexing='ij'), -1).reshape(-1, 2)
        select_inds = np.random.choice(coords.shape[0], size=[CONFIG['batch_size']], replace=False)
        select_inds = torch.from_numpy(select_inds).long().to(device)
        
        b_o = rays_o.reshape(-1, 3)[select_inds]
        b_d = rays_d.reshape(-1, 3)[select_inds]
        b_r = radii.reshape(-1, 1)[select_inds]
        target_rgb = target.reshape(-1, 3)[select_inds]
        
        near, far = bds.min() * 0.9, bds.max() * 1.0
        t_vals = torch.linspace(near, far, CONFIG['N_samples'] + 1, device=device).expand(b_o.shape[0], CONFIG['N_samples'] + 1)
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_vals = lower + (upper - lower) * torch.rand(t_vals.shape, device=device)
        
        t0 = t_vals[..., :-1]; t1 = t_vals[..., 1:]
        
        means, covs = conical_frustum_to_gaussian(b_o[:,None,:], b_d[:,None,:], t0, t1, b_r)
        viewdirs = b_d / torch.norm(b_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(means.shape)
        
        raw = model(means, covs, viewdirs)
        rgb_map, _, _ = raw2outputs_mip(raw, (t0+t1)/2, b_d)
        
        loss = torch.mean((rgb_map - target_rgb) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())

        if i % CONFIG['i_val'] == 0 and i > 0:
            model.eval()
            with torch.no_grad():
                target_val = images[val_idx]
                pose_val = poses[val_idx, :3, :4]
                
                rays_o_val, rays_d_val, radii_val = get_rays_mip(H, W, K, pose_val)
                
                rgb_val = render_full_image_mip(model, rays_o_val, rays_d_val, radii_val, near, far, chunk=CONFIG['batch_size'])
                
                mse = torch.mean((rgb_val - target_val) ** 2)
                psnr = -10. * torch.log10(mse)
                psnr_val = psnr.item()
                
                val_psnr_history.append(psnr_val)
                iter_history.append(i)
                
                desc = f"PSNR: {psnr_val:.2f}"
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    patience_counter = 0
                    torch.save(model.state_dict(), f"./logs/{CONFIG['expname']}_BEST.pth")
                else:
                    patience_counter += 1
                
                pbar.set_description(desc)
                
                if patience_counter >= CONFIG['patience']:
                    print(f"Early Stopping! Best PSNR: {best_psnr}")
                    break
            model.train()

    torch.save(model.state_dict(), f"./logs/{CONFIG['expname']}_LAST.pth")
    
    metrics = {
        'loss': train_loss_history,
        'psnr': val_psnr_history,
        'iters': iter_history,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    with open(f"./logs/{CONFIG['expname']}_metrics.json", 'w') as f:
        json.dump(metrics, f)
        
    print("Entrenamiento Mip-NeRF finalizado.")

if __name__ == '__main__':
    train()
