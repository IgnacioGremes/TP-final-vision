import torch
import numpy as np
import os
import tqdm
import json  
import matplotlib.pyplot as plt 
from nerf_pytorch import Embedder, FastNeRF, raw2outputs, get_rays
from load_llff import load_llff_data 

CONFIG = {
    'expname': 'pesos_modelo', 
    'datadir': './data/nerf_llff_data/fern',
    'factor': 4,      
    'N_samples': 128,
    'N_iters': 10000,          
    'batch_size': 4096,
    'lrate': 5e-4,
    'i_val': 500,
    'layers': 8,
    'neurons': 256 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_full_image(model, rays_o, rays_d, near, far, embed_pts, embed_views, chunk=32768):
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    rgb_frames = []
    
    for i in range(0, rays_o.shape[0], chunk):
        batch_rays_o = rays_o[i:i+chunk]
        batch_rays_d = rays_d[i:i+chunk]
        
        z_vals = torch.linspace(near, far, 64, device=device).expand(batch_rays_o.shape[0], 64)
        pts = batch_rays_o[...,None,:] + batch_rays_d[...,None,:] * z_vals[...,:,None]
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(pts.shape)
        
        embedded = torch.cat([embed_pts.embed(pts.reshape(-1,3)), embed_views.embed(viewdirs.reshape(-1,3))], -1)
        
        with torch.no_grad():
            raw = model(embedded).reshape(batch_rays_o.shape[0], 64, 4)
            rgb_map, _, _ = raw2outputs(raw, z_vals, batch_rays_d)
            
        rgb_frames.append(rgb_map)
        
    return torch.cat(rgb_frames, 0).reshape(H, W, 3)

def train():
    print("Cargando datos...")
    images, poses, bds, render_poses, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'],
        recenter=True, bd_factor=.75, spherify=False)
    
    H, W = images.shape[1:3]
    original_H = poses[0, 0, -1]
    focal = poses[0, 2, -1]
    if H != original_H:
        focal *= (H / original_H)
    H, W = int(H), int(W)
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    bds = torch.Tensor(bds).to(device)

    if np.ndim(i_test) == 0:
        val_idx = int(i_test)
    else:
        val_idx = int(i_test[0])

    test_idx = 7 
    
    if test_idx == val_idx: 
        test_idx = (val_idx + 1) % int(images.shape[0])

    print(f"--- CONFIGURACIÓN DE SPLITS ---")
    print(f"Total Imágenes: {images.shape[0]}")
    print(f"Validación (Gráfico): Imagen #{val_idx}")
    print(f"Test Puro (Final):    Imagen #{test_idx}")

    all_indices = np.arange(int(images.shape[0]))
    exclude_list = [val_idx, test_idx]
    i_train = np.array([i for i in all_indices if i not in exclude_list])

    print(f"Entrenando con {len(i_train)} imágenes: {i_train}")

    embed_pts = Embedder(input_dims=3, num_freqs=10)
    embed_views = Embedder(input_dims=3, num_freqs=4)
    model = FastNeRF(D=CONFIG['layers'], W=CONFIG['neurons']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lrate'])

    train_loss_history = []
    val_psnr_history = []
    iter_history = []

    print("Iniciando entrenamiento...")
    pbar = tqdm.tqdm(range(CONFIG['N_iters']))
    
    for i in pbar:
        img_idx = np.random.choice(i_train)
        target_img = images[img_idx]
        pose = poses[img_idx, :3, :4]
        rays_o, rays_d = get_rays(H, W, K, pose)
        
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H, device=device), torch.linspace(0, W-1, W, device=device), indexing='ij'), -1).reshape(-1, 2)
        select_inds = np.random.choice(coords.shape[0], size=[CONFIG['batch_size']], replace=False)
        select_inds = torch.from_numpy(select_inds).long().to(device)
        batch_rays_o = rays_o.reshape(-1, 3)[select_inds]
        batch_rays_d = rays_d.reshape(-1, 3)[select_inds]
        target_rgb = target_img.reshape(-1, 3)[select_inds]
        
        near, far = bds.min() * 0.9, bds.max() * 1.0
        z_vals = torch.linspace(near, far, CONFIG['N_samples'], device=device).expand(batch_rays_o.shape[0], CONFIG['N_samples'])
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
        
        pts = batch_rays_o[...,None,:] + batch_rays_d[...,None,:] * z_vals[...,:,None]
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(pts.shape)
        
        embedded = torch.cat([embed_pts.embed(pts.reshape(-1,3)), embed_views.embed(viewdirs.reshape(-1,3))], -1)
        raw = model(embedded).reshape(CONFIG['batch_size'], CONFIG['N_samples'], 4)
        rgb_map, _, _ = raw2outputs(raw, z_vals, batch_rays_d)
        
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
                rays_o_val, rays_d_val = get_rays(H, W, K, pose_val)
                
                rgb_val = render_full_image(model, rays_o_val, rays_d_val, near, far, embed_pts, embed_views)
                
                mse = torch.mean((rgb_val - target_val) ** 2)
                psnr = -10. * torch.log10(mse)
                
                val_psnr_history.append(psnr.item())
                iter_history.append(i)
                pbar.set_description(f"PSNR: {psnr.item():.2f}")
            model.train()

    os.makedirs('./logs', exist_ok=True)
    torch.save(model.state_dict(), f"./logs/{CONFIG['expname']}.pth")
    
    metrics = {
        'loss': train_loss_history,
        'psnr': val_psnr_history,
        'iters': iter_history,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    with open(f"./logs/modelo_metrics.json", 'w') as f:
        json.dump(metrics, f)
        
    print("Entrenamiento finalizado. Métricas guardadas en json.")

if __name__ == '__main__':
    train()
