import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from nerf_pytorch import Embedder, FastNeRF, raw2outputs, get_rays
from load_llff import load_llff_data

# CONFIGURACIÓN (Asegúrate que coincida con tu entrenamiento)
CONFIG = {
    'model_path': './logs/fern_final_hd.pth', 
    'datadir': './data/nerf_llff_data/fern',
    'factor': 8, 
    'N_samples': 64, 
    'chunk': 32768
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_image_compare(model, rays_o, rays_d, near, far, embed_pts, embed_views):
    """ Función auxiliar para renderizar una sola imagen por partes (chunks) """
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    rgb_frames = []
    
    for i in range(0, rays_o.shape[0], CONFIG['chunk']):
        batch_rays_o = rays_o[i:i+CONFIG['chunk']]
        batch_rays_d = rays_d[i:i+CONFIG['chunk']]
        
        z_vals = torch.linspace(near, far, CONFIG['N_samples']).to(device)
        z_vals = z_vals.expand(batch_rays_o.shape[0], CONFIG['N_samples'])
        
        pts = batch_rays_o[...,None,:] + batch_rays_d[...,None,:] * z_vals[...,:,None]
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(pts.shape)
        
        embedded = torch.cat([embed_pts.embed(pts.reshape(-1,3)), embed_views.embed(viewdirs.reshape(-1,3))], -1)
        
        with torch.no_grad():
            raw = model(embedded).reshape(batch_rays_o.shape[0], CONFIG['N_samples'], 4)
            rgb_map, _, _ = raw2outputs(raw, z_vals, batch_rays_d)
            
        rgb_frames.append(rgb_map.cpu().numpy())
        
    return np.concatenate(rgb_frames, 0).reshape(H, W, 3)

def run_compare():
    print("--- GENERANDO COMPARACIÓN ---")
    
    # 1. Cargar datos
    images, poses, bds, _, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'], recenter=True, bd_factor=.75, spherify=False)
    
    # --- CORRECCIÓN DE DIMENSIONES (Igual que en train) ---
    H, W = images.shape[1:3]
    original_H = poses[0, 0, -1]
    focal = poses[0, 2, -1]
    if H != original_H:
        focal *= (H / original_H)
    H, W = int(H), int(W)
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    # -----------------------------------------------------

    # 2. Cargar Modelo
    embed_pts = Embedder(3, 10); embed_views = Embedder(3, 4)
    model = FastNeRF(4, 128).to(device)
    
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
        print("Modelo cargado.")
    else:
        print(f"⚠️ ERROR: No encuentro {CONFIG['model_path']}. Verifica la ruta.")
        return

    model.eval()

    # 3. Seleccionar imagen de prueba (CORRECCIÓN DEL ERROR)
    # Si i_test es un escalar de numpy (ndim=0) o int, lo usamos directo.
    # Si es un array, tomamos el primero.
    if np.ndim(i_test) == 0:
        idx = int(i_test)
    else:
        idx = i_test[0]
        
    print(f"Renderizando imagen de prueba #{idx}...")
    
    target = images[idx]
    pose = torch.Tensor(poses[idx, :3, :4]).to(device)
    near, far = bds.min() * 0.9, bds.max() * 1.0

    # 4. Renderizar
    rays_o, rays_d = get_rays(H, W, K, pose)
    rgb_pred = render_image_compare(model, rays_o, rays_d, near, far, embed_pts, embed_views)

    # 5. Guardar Comparación
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.title(f"Real (Imagen #{idx})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(rgb_pred, 0, 1))
    plt.title("Predicción NeRF")
    plt.axis('off')

    out_path = "./logs/comparacion_fern.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"¡Comparación guardada en {out_path}!")

if __name__ == '__main__':
    run_compare()
