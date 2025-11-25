import torch
import numpy as np
import os
import tqdm
from nerf_pytorch import Embedder, FastNeRF, raw2outputs, get_rays
from load_llff import load_llff_data 

# --- CONFIGURACIÓN OPTIMIZADA ---
CONFIG = {
    'expname': 'fern_fast',
    'datadir': './data/nerf_llff_data/fern',
    'factor': 8,       # Downsample 8x (imágenes pequeñas = MUY RÁPIDO)
    'N_samples': 64,   # Pocas muestras por rayo
    'N_iters': 2000,   # Pocas iteraciones para probar que funciona (luego sube a 50000)
    'batch_size': 4096,
    'lrate': 5e-4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- USANDO DISPOSITIVO: {device} ---")

def train():
    # 1. Cargar Datos
    images, poses, bds, render_poses, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'],
        recenter=True, bd_factor=.75, spherify=False)
    
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    bds = torch.Tensor(bds).to(device)
    
    H, W, focal = poses[0, :3, -1]
    H, W = int(H), int(W)
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    
    print(f"Datos cargados. Imágenes: {images.shape}")

    i_test = [i_test] if isinstance(i_test, int) else i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if i not in i_test])

    # 2. Modelo y Embedders
    embed_pts = Embedder(input_dims=3, num_freqs=10)
    embed_views = Embedder(input_dims=3, num_freqs=4)
    model = FastNeRF(D=4, W=128).to(device) # Red pequeña
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lrate'])

    # 3. Entrenamiento
    pbar = tqdm.tqdm(range(CONFIG['N_iters']))
    for i in pbar:
        img_idx = np.random.choice(i_train)
        target_img = images[img_idx]
        pose = poses[img_idx, :3, :4]
        
        rays_o, rays_d = get_rays(H, W, K, pose)
        
        # Sampling aleatorio de píxeles (Batching)
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1).reshape(-1, 2)
        select_inds = np.random.choice(coords.shape[0], size=[CONFIG['batch_size']], replace=False)
        select_inds = torch.from_numpy(select_inds).long().to(device)
        
        batch_rays_o = rays_o.reshape(-1, 3)[select_inds]
        batch_rays_d = rays_d.reshape(-1, 3)[select_inds]
        target_rgb = target_img.reshape(-1, 3)[select_inds]
        
        # Integración a lo largo del rayo
        near, far = bds.min() * 0.9, bds.max() * 1.0
        t_vals = torch.linspace(0., 1., CONFIG['N_samples']).to(device)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        # Perturbación para evitar overfitting
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.cat([mids, z_vals[-1:]], -1)
        lower = torch.cat([z_vals[:1], mids], -1)
        t_rand = torch.rand(CONFIG['batch_size'], CONFIG['N_samples']).to(device)
        z_vals = lower + (upper - lower) * t_rand
        
        # Preparar inputs (Posición + Dirección)
        pts = batch_rays_o[...,None,:] + batch_rays_d[...,None,:] * z_vals[...,:,None]
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(pts.shape)
        
        pts_flat = pts.reshape(-1, 3)
        views_flat = viewdirs.reshape(-1, 3)
        
        embedded = torch.cat([embed_pts.embed(pts_flat), embed_views.embed(views_flat)], -1)
        
        # Forward Pass
        raw = model(embedded)
        raw = raw.reshape(CONFIG['batch_size'], CONFIG['N_samples'], 4)
        
        rgb_map, _, _ = raw2outputs(raw, z_vals, batch_rays_d)
        
        # Loss
        loss = torch.mean((rgb_map - target_rgb) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            psnr = -10. * torch.log10(loss)
            pbar.set_description(f"PSNR: {psnr.item():.2f}")

    # Guardar modelo final
    os.makedirs('./logs', exist_ok=True)
    torch.save(model.state_dict(), f"./logs/{CONFIG['expname']}_final.pth")
    print("Entrenamiento finalizado y guardado.")

if __name__ == '__main__':
    train()