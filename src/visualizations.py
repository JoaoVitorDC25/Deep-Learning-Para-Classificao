import matplotlib.pyplot as plt
import numpy as np

# Função para mostrar a imagem
def imshow(img):
    img = img / 2 + 0.5                          # Desnormaliza (de [-1, 1] para [0, 1])
    npimg = img.numpy()                          # Converte a imagem em formato de matriz NumPy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))   # Converte de (C, H, W) para (H, W, C) - Canais (C), Altura (H) e Largura (W)
    plt.show()