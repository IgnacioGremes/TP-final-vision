import torch
import numpy as np
import imageio
import tqdm
import os
from nerf_pytorch import Embedder, FastNeRF, raw2outputs, get_rays
from load_llff import load_llff_data

# --- CONFIGURACIÓN (Debe coincidir con el entrenamiento) ---
CONFIG = {
    'model_path': './logs/fern_fast_final.pth', # <--- ASEGURA QUE ESTE NOMBRE COINCIDA CON TU ARCHIVO GUARDADO
    'datadir': './data/nerf_llff_data/fern',
    'factor': 8,       
    'N_samples': 64,
    'chunk': 16384,     # Cantidad de rayos a procesar a la vez (para no llenar la memoria)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- RENDERIZANDO EN: {device} ---")

def render_image(model, rays_o, rays_d, near, far, embed_pts, embed_views):
    """Renderiza una imagen completa procesando rayos por lotes (chunks)"""
    H, W = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    rgb_frames = []
    
    # Procesar por partes para no explotar la GPU
    for i in range(0, rays_o.shape[0], CONFIG['chunk']):
        batch_rays_o = rays_o[i:i+CONFIG['chunk']]
        batch_rays_d = rays_d[i:i+CONFIG['chunk']]
        
        z_vals = torch.linspace(near, far, CONFIG['N_samples']).to(device)
        # Broadcasting para crear [batch, samples]
        z_vals = z_vals.expand(batch_rays_o.shape[0], CONFIG['N_samples'])
        
        # Puntos en el espacio
        pts = batch_rays_o[...,None,:] + batch_rays_d[...,None,:] * z_vals[...,:,None]
        
        # Direcciones de vista
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(pts.shape)
        
        # Flatten para la red
        pts_flat = pts.reshape(-1, 3)
        views_flat = viewdirs.reshape(-1, 3)
        
        embedded = torch.cat([embed_pts.embed(pts_flat), embed_views.embed(views_flat)], -1)
        
        # Inferencia
        with torch.no_grad():
            raw = model(embedded)
            raw = raw.reshape(batch_rays_o.shape[0], CONFIG['N_samples'], 4)
            rgb_map, _, _ = raw2outputs(raw, z_vals, batch_rays_d)
            
        rgb_frames.append(rgb_map.cpu().numpy())
        
    return np.concatenate(rgb_frames, 0).reshape(H, W, 3)

def run_render():
    print("--- INICIANDO RENDERIZADO DE VIDEO ---")
    
    # 1. Cargar Datos (Solo poses)
    _, poses, bds, render_poses, _ = load_llff_data(
        CONFIG['datadir'], CONFIG['factor'],
        recenter=True, bd_factor=.75, spherify=False)
    
    # --- CORRECCIÓN CRÍTICA DE RESOLUCIÓN ---
    # Obtenemos las dimensiones originales de los metadatos
    H_orig = poses[0, 0, -1]
    W_orig = poses[0, 1, -1]
    focal_orig = poses[0, 2, -1]
    
    # Forzamos la reducción según el factor de configuración (8)
    # Esto asegura que rendericemos en 378x504 y no en 3024x4032
    H = int(H_orig / CONFIG['factor'])
    W = int(W_orig / CONFIG['factor'])
    focal = focal_orig / CONFIG['factor']
    
    print(f"Resolución Corregida: {H}x{W} (Focal: {focal:.2f})")
    # ----------------------------------------
    
    K = torch.Tensor([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]).to(device)
    near, far = bds.min() * 0.9, bds.max() * 1.0

    # 2. Cargar Modelo
    embed_pts = Embedder(input_dims=3, num_freqs=10)
    embed_views = Embedder(input_dims=3, num_freqs=4)
    model = FastNeRF(D=4, W=128).to(device)
    
    # Cargar pesos guardados
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
        print("Modelo cargado exitosamente.")
    else:
        print(f"ERROR: No se encuentra el modelo en {CONFIG['model_path']}")
        # Intentar buscar cualquier .pth en logs
        try:
            fallback = [f for f in os.listdir('./logs') if f.endswith('.pth')][0]
            fallback_path = os.path.join('./logs', fallback)
            print(f"Usando fallback: {fallback_path}")
            model.load_state_dict(torch.load(fallback_path, map_location=device))
        except:
            return

    model.eval()
    
    # 3. Renderizar Frames
    frames = []
    # Render_poses es una lista de matrices 3x5 generada por load_llff (trayectoria espiral)
    print(f"Generando {len(render_poses)} frames...")
    
    for i, c2w in enumerate(tqdm.tqdm(render_poses)):
        c2w = torch.Tensor(c2w[:3, :4]).to(device)
        
        rays_o, rays_d = get_rays(H, W, K, c2w)
        
        rgb = render_image(model, rays_o, rays_d, near, far, embed_pts, embed_views)
        
        # Convertir a formato de imagen (0-255 uint8)
        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        frames.append(rgb8)

    # 4. Guardar Video
    output_path = './logs/video_resultado.mp4'
    imageio.mimwrite(output_path, frames, fps=30, quality=8)
    print(f"¡VIDEO GUARDADO EN! {output_path}")

if __name__ == '__main__':
    run_render()
