import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from nerf_pytorch import Embedder, FastNeRF, raw2outputs, get_rays
from load_llff import load_llff_data

# --- CONFIGURACIÓN ---
CONFIG = {
    'model_path': './logs/pesos_modelo.pth',  # <--- Asegúrate que este sea tu modelo entrenado
    'datadir': './data/nerf_llff_data/fern',
    'factor': 4, 
    'N_samples': 128, 
    'chunk': 32768,
    'layers': 8,
    'neurons': 256
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_image_compare(model, rays_o, rays_d, near, far, embed_pts, embed_views):
    """ Renderiza una imagen completa por partes """
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    rgb_frames = []
    
    for i in range(0, rays_o.shape[0], CONFIG['chunk']):
        batch_rays_o = rays_o[i:i+CONFIG['chunk']]
        batch_rays_d = rays_d[i:i+CONFIG['chunk']]
        
        z_vals = torch.linspace(near, far, CONFIG['N_samples'], device=device).expand(batch_rays_o.shape[0], CONFIG['N_samples'])
        pts = batch_rays_o[...,None,:] + batch_rays_d[...,None,:] * z_vals[...,:,None]
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(pts.shape)
        
        embedded = torch.cat([embed_pts.embed(pts.reshape(-1,3)), embed_views.embed(viewdirs.reshape(-1,3))], -1)
        
        with torch.no_grad():
            raw = model(embedded).reshape(batch_rays_o.shape[0], CONFIG['N_samples'], 4)
            rgb_map, _, _ = raw2outputs(raw, z_vals, batch_rays_d)
            
        rgb_frames.append(rgb_map.cpu().numpy())
        
    return np.concatenate(rgb_frames, 0).reshape(H, W, 3)

def save_comparison(target, pred, title, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.title(f"Real (Ground Truth)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(pred, 0, 1))
    plt.title(f"Predicción NeRF")
    plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Guardado: {filename}")

def run_compare():
    print(f"--- GENERANDO COMPARACIONES PARA {CONFIG['expname'] if 'expname' in CONFIG else 'MODELO'} ---")
    
    # 1. Cargar datos
    images, poses, bds, _, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'], recenter=True, bd_factor=.75, spherify=False)
    
    # Manejo de índices train/test
    if not isinstance(i_test, (list, np.ndarray)): i_test = [i_test]
    val_idx = int(i_test[0])
    test_idx = 7  # <--- La imagen secreta
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if i not in i_test])

    # Ajuste de resolución y focal
    H, W = images.shape[1:3]
    original_H = poses[0, 0, -1]
    focal = poses[0, 2, -1]
    if H != original_H: focal *= (H / original_H)
    H, W = int(H), int(W)
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    near, far = bds.min() * 0.9, bds.max() * 1.0

    # 2. Cargar Modelo
    embed_pts = Embedder(3, 10); embed_views = Embedder(3, 4)
    model = FastNeRF(CONFIG['layers'], CONFIG['neurons']).to(device)
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    else:
        print("❌ ERROR: Modelo no encontrado."); return
    model.eval()

    # 3. Definir qué imágenes vamos a comparar
    # Seleccionamos 2 de train (ej. la primera y una del medio) y todas las de test
    indices_to_plot = {
        'Train_View': i_train[0],       # Una que ya vio (debe verse perfecta)
        'Validation_View': val_idx,     # La del gráfico (debe verse muy bien)
        'Test_View': test_idx      # La prueba de fuego (nunca vista)
    }
    
    # Si hay más de una imagen de test, agregamos otra
    if len(i_test) > 1:
        indices_to_plot['Test_View_1'] = i_test[1]

    # 4. Loop de generación
    for name, idx in indices_to_plot.items():
        print(f"Renderizando {name} (Imagen #{idx})...")
        target = images[idx]
        pose = torch.Tensor(poses[idx, :3, :4]).to(device)
        
        rays_o, rays_d = get_rays(H, W, K, pose)
        rgb_pred = render_image_compare(model, rays_o, rays_d, near, far, embed_pts, embed_views)
        
        save_comparison(target, rgb_pred, f"{name} (Img #{idx})", f"./logs/compare_{name}.png")

if __name__ == '__main__':
    run_compare()
