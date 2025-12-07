import json
import matplotlib.pyplot as plt
import os

FILES_TO_PLOT = {
    'output/mip/fern_mipnerf_final_metrics.json': 'Mip-NeRF (Fern)',
    'output/10/modelo_metrics_10_horns.json':     'FastNeRF (Horns)',
    'output/9/modelo_metrics_9.json':             'FastNeRF (Fern)'
}

OUTPUT_IMAGE = 'comparacion_psnr.png'

def plot_multigraph():
    plt.figure(figsize=(10, 6))
    
    styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    for i, (file_path, label) in enumerate(FILES_TO_PLOT.items()):

        if not os.path.exists(file_path):
            print(f"⚠️ AVISO: No se encontró el archivo: {file_path}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            psnr = data.get('psnr', [])
            iters = data.get('iters', [])
            
            if not psnr or not iters:
                print(f"El archivo {file_path} no tiene datos de 'psnr' o 'iters'.")
                continue

            plt.plot(iters, psnr, 
                     linestyle=styles[i % len(styles)], 
                     marker=markers[i % len(markers)], 
                     markersize=4, 
                     label=label,
                     alpha=0.8,
                     linewidth=2)
            
            print(f"Cargado: {label} ({len(psnr)} puntos)")
            
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")


    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('PSNR de Validación (dB)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n✨ Gráfico comparativo guardado en: {OUTPUT_IMAGE}")


if __name__ == '__main__':
    plot_multigraph()
