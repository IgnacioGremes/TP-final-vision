import json
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURACIÓN ---
# Asegúrate de que este nombre coincida con el que usaste en train_nerf.py
JSON_PATH = f'./logs/modelo_metrics.json'
OUTPUT_PATH = f'./logs/metric_curves.png'

def moving_average(data, window_size=100):
    """Suaviza la curva de Loss para que se vea más limpia"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_graphs():
    # 1. Verificar que el archivo existe
    if not os.path.exists(JSON_PATH):
        print(f"❌ ERROR: No se encuentra el archivo {JSON_PATH}")
        print("Asegúrate de que el entrenamiento haya terminado y creado el JSON.")
        return

    # 2. Cargar datos
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    loss_data = data['loss']
    psnr_data = data['psnr']
    iters_psnr = data['iters']
    
    print(f"Datos cargados: {len(loss_data)} iteraciones de loss, {len(psnr_data)} puntos de PSNR.")

    # 3. Configurar la figura (2 gráficos lado a lado)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- GRÁFICO 1: LOSS (Pérdida) ---
    # Graficamos los datos crudos con transparencia
    ax1.plot(loss_data, alpha=0.3, color='gray', label='Raw Loss')
    
    # Graficamos una media móvil (suavizada) si hay suficientes datos
    if len(loss_data) > 100:
        smooth_loss = moving_average(loss_data, window_size=100)
        # Ajustar eje X para la media móvil
        ax1.plot(range(99, len(loss_data)), smooth_loss, color='red', linewidth=2, label='Smoothed Loss (Avg 100)')
    
    ax1.set_title('Training Loss (MSE)')
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log') # Escala logarítmica suele verse mejor para Loss
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # --- GRÁFICO 2: PSNR (Calidad) ---
    ax2.plot(iters_psnr, psnr_data, marker='o', color='blue', linewidth=2, markersize=4, label='Validation PSNR')
    
    ax2.set_title('Validation PSNR (Higher is Better)')
    ax2.set_xlabel('Iteraciones')
    ax2.set_ylabel('PSNR (dB)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    # 4. Guardar y Mostrar
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"✅ Gráficos guardados exitosamente en: {OUTPUT_PATH}")
    # plt.show() # Descomentar si lo corres en tu PC local, en Colab no hace falta

if __name__ == '__main__':
    plot_graphs()
