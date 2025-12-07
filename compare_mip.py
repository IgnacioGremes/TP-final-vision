import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from mipnerf_pytorch import MipNeRF, conical_frustum_to_gaussian, raw2outputs_mip
from load_llff import load_llff_data

CONFIG = {
    'model_path': './logs/fern_mipnerf_final_BEST.pth', 
    'datadir': './data/nerf_llff_data/fern',
    'factor': 4, 
    'N_samples': 128, 
    'chunk': 4096 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rays_mip_compare(H, W, K, c2w):
    """ Genera rayos y calcula sus radios base para Mip-NeRF """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=c2w.device), 
                          torch.linspace(0, H-1, H, device=c2w.device), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    
    # Radio base del cono
    base_radius = (2 / np.sqrt(12)) / (2 * K[0,0]) 
    radii = torch.ones_like(rays_d[..., :1]) * base_radius
    
    return rays_o, rays_d, radii

def render_image_mip(model, rays_o, rays_d, radii, near, far):
    """ Renderiza la imagen completa por chunks usando lógica de conos """
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3); rays_d = rays_d.reshape(-1, 3); radii = radii.reshape(-1, 1)
    rgb_frames = []
    
    for i in range(0, rays_o.shape[0], CONFIG['chunk']):
        b_o = rays_o[i:i+CONFIG['chunk']]; b_d = rays_d[i:i+CONFIG['chunk']]; b_r = radii[i:i+CONFIG['chunk']]
        
        # Intervalos
        t_vals = torch.linspace(near, far, CONFIG['N_samples'] + 1, device=device).expand(b_o.shape[0], CONFIG['N_samples'] + 1)
        t0 = t_vals[..., :-1]; t1 = t_vals[..., 1:]
        
        # Convertir a Gaussianas
        means, covs = conical_frustum_to_gaussian(b_o[:,None,:], b_d[:,None,:], t0, t1, b_r)
        
        viewdirs = b_d / torch.norm(b_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(means.shape)
        
        with torch.no_grad():
            raw = model(means, covs, viewdirs)
            rgb, _, _ = raw2outputs_mip(raw, (t0+t1)/2, b_d)
            
        rgb_frames.append(rgb.cpu().numpy())
        
    return np.concatenate(rgb_frames, 0).reshape(H, W, 3)

def save_comparison(target, pred, title, filename):
    plt.figure(figsize=(10, 5))
    
    # Imagen Real
    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.title("Real (Ground Truth)")
    plt.axis('off')

    # Imagen Predicha
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(pred, 0, 1))
    plt.title("Predicción Mip-NeRF")
    plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Guardado: {filename}")

def run_compare():
    print(f"--- COMPARACIÓN MIP-NERF ({device}) ---")
    
    # Cargar Datos
    images, poses, bds, _, i_test = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'], recenter=True, bd_factor=.75, spherify=False)
    
    H, W = images.shape[1:3]
    original_H = poses[0, 0, -1]
    focal = poses[0, 2, -1]
    if H != original_H: focal *= (H / original_H)
    H, W = int(H), int(W)
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    near, far = bds.min() * 0.9, bds.max() * 1.0

    # Cargar Modelo
    model = MipNeRF(D=8, W=256).to(device)
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
        print("Modelo cargado exitosamente.")
    else:
        print(f"ERROR: No encuentro el archivo {CONFIG['model_path']}")
        return
    model.eval()

    # Entrenamos excluyendo 12 y 7.
    indices_to_plot = {
        'Train_View': 0,
        'Validation_View': 12,
        'TEST_PURO_View': 7
    }

    # Renderizar y Guardar
    for name, idx in indices_to_plot.items():
        print(f"Renderizando {name} (Imagen #{idx})...")
        
        target = images[idx]
        pose = torch.Tensor(poses[idx, :3, :4]).to(device)
        
        rays_o, rays_d, radii = get_rays_mip_compare(H, W, K, pose)
        
        rgb_pred = render_image_mip(model, rays_o, rays_d, radii, near, far)
        
        save_comparison(target, rgb_pred, f"Mip-NeRF: {name} (Img #{idx})", f"./logs/compare_mip_{name}.png")

if __name__ == '__main__':
    run_compare()
