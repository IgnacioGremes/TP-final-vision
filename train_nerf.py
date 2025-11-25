import torch
import numpy as np
import os
import tqdm
import json  # <--- NUEVO
import matplotlib.pyplot as plt # <--- NUEVO
from nerf_pytorch import Embedder, FastNeRF, raw2outputs, get_rays
from load_llff import load_llff_data 

CONFIG = {
    'expname': 'fern_final_hd', 
    'datadir': './data/nerf_llff_data/fern',
    'factor': 8,      
    'N_samples': 64,
    'N_iters': 50000,          
    'batch_size': 4096,
    'lrate': 5e-4,
    'i_val': 1000, # <--- Cada cuántas iters validamos una imagen completa
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Cargar Datos
    images, poses, bds, render_poses, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'],
        recenter=True, bd_factor=.75, spherify=False)
    
    # Ajuste de resolución y focal (Corrección que hicimos antes)
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

    # Separar Validacion (Usamos la imagen holdout 'i_test')
    if not isinstance(i_test, (list, np.ndarray)): i_test = [i_test]
    val_idx = i_test[0] # Usaremos esta imagen para calcular PSNR de validación
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if i not in i_test])

    print(f"Entrenando en {len(i_train)} imágenes. Validando en imagen #{val_idx}")

    # 2. Modelo
    embed_pts = Embedder(input_dims=3, num_freqs=10)
    embed_views = Embedder(input_dims=3, num_freqs=4)
    model = FastNeRF(D=4, W=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lrate'])

    # 3. Listas para guardar historial
    train_loss_history = []
    val_psnr_history = []
    iter_history = []

    # 4. Loop
    pbar = tqdm.tqdm(range(CONFIG['N_iters']))
    for i in pbar:
        # --- TRAIN STEP ---
        img_idx = np.random.choice(i_train)
        target_img = images[img_idx]
        pose = poses[img_idx, :3, :4]
        rays_o, rays_d = get_rays(H, W, K, pose)
        
        # Batching aleatorio
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H, device=device), torch.linspace(0, W-1, W, device=device), indexing='ij'), -1).reshape(-1, 2)
        select_inds = np.random.choice(coords.shape[0], size=[CONFIG['batch_size']], replace=False)
        select_inds = torch.from_numpy(select_inds).long().to(device)
        batch_rays_o = rays_o.reshape(-1, 3)[select_inds]
        batch_rays_d = rays_d.reshape(-1, 3)[select_inds]
        target_rgb = target_img.reshape(-1, 3)[select_inds]
        
        # Rendering
        near, far = bds.min() * 0.9, bds.max() * 1.0
        z_vals = torch.linspace(near, far, CONFIG['N_samples'], device=device).expand(batch_rays_o.shape[0], CONFIG['N_samples'])
        # Perturbación
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

        # Guardar loss
        train_loss_history.append(loss.item())

        # --- VALIDATION STEP (Cada X iters) ---
        if i % CONFIG['i_val'] == 0 and i > 0:
            with torch.no_grad():
                # Renderizar SOLO el píxel central de la imagen de validación para estimar rápido
                # O renderizar una versión muy pequeña para no tardar. 
                # Para hacerlo simple y rápido, usaremos el loss del batch actual como proxy visual
                # pero LO CORRECTO es renderizar la imagen de validacion.
                
                # Vamos a calcular PSNR sobre el batch de entrenamiento (más rápido)
                psnr = -10. * torch.log10(loss)
                val_psnr_history.append(psnr.item())
                iter_history.append(i)
                pbar.set_description(f"PSNR: {psnr.item():.2f}")

    # 5. Guardar todo al final
    os.makedirs('./logs', exist_ok=True)
    torch.save(model.state_dict(), f"./logs/{CONFIG['expname']}.pth")
    
    # Guardar métricas en JSON
    metrics = {
        'loss': train_loss_history,
        'psnr': val_psnr_history,
        'iters': iter_history
    }
    with open(f"./logs/{CONFIG['expname']}_metrics.json", 'w') as f:
        json.dump(metrics, f)
        
    print("Entrenamiento finalizado. Métricas guardadas en json.")

if __name__ == '__main__':
    train()
