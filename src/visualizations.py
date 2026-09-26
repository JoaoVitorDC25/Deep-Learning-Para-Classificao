import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Função para mostrar a imagem
def imshow(img):
    img = img / 2 + 0.5                          # Desnormaliza (de [-1, 1] para [0, 1])
    npimg = img.numpy()                          # Converte a imagem em formato de matriz NumPy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))   # Converte de (C, H, W) para (H, W, C) - Canais (C), Altura (H) e Largura (W)
    plt.title("Imagem de amostra do conjunto de teste")
    
    # Diretório onde as imagens classificadas serão salvas
    output_dir = Path("../images/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo de saída
    output_path = output_dir / f"image_sample.png"
    
    # Salva a figura
    plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()