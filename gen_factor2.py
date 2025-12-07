import os
from PIL import Image
import sys

SCENES = ['fern', 'flower']
BASE_DIR = './data/nerf_llff_data'
FACTOR = 2

def generate_res():
    for scene in SCENES:
        src_dir = os.path.join(BASE_DIR, scene, 'images')
        dst_dir = os.path.join(BASE_DIR, scene, f'images_{FACTOR}')
        
        if not os.path.exists(src_dir):
            print(f"X No encuentro la carpeta original: {src_dir}")
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Generando factor {FACTOR} para escena '{scene}'...")
        
        images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.JPG', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(src_dir, img_name)
            img = Image.open(img_path)
            
            W, H = img.size
            new_W, new_H = W // FACTOR, H // FACTOR
            
            img_resized = img.resize((new_W, new_H), Image.Resampling.LANCZOS)
            
            out_path = os.path.join(dst_dir, img_name)
            img_resized.save(out_path)
            
        print(f"OK ¡Listo! Imágenes guardadas en {dst_dir}")

if __name__ == '__main__':
    generate_res()
