import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

# Classe para criar a arquitetura do modelo
class ConvNet(nn.Module):

    # Método construtor
    def __init__(self):

        # Inicializa o construtor da classe mãe
        super(ConvNet, self).__init__()

        # Input: 3 canais de cor, Output: 6 feature maps, Kernel: 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Max pooling 2x2
        self.pool = nn.MaxPool2d(2, 2)

        # Input: 6 canais (do conv1), Output: 16 feature maps, Kernel: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # As imagens CIFAR-10 são 32x32
        # Após conv1 (5x5): 32-5+1 = 28 -> 28x28
        # Após pool1 (2x2): 28/2 = 14 -> 14x14
        # Após conv2 (5x5): 14-5+1 = 10 -> 10x10
        # Após pool2 (2x2): 10/2 = 5 -> 5x5
        # Tamanho achatado (flattened): 16 canais * 5 * 5 = 400 atributos

        # Camadas totalmente conectadas (Linear)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 400 -> 120
        self.fc2 = nn.Linear(120, 84)         # 120 -> 84
        self.fc3 = nn.Linear(84, 10)          # 84 -> 10 (10 classes)

    # Método forward (passada para a frente)
    def forward(self, x):

        # Aplicando as camadas convolucionais e pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Achatar (flatten) o tensor para a camada linear
        # Achata todas as dimensões, exceto o batch
        x = torch.flatten(x, 1)

        # Aplicando as camadas lineares com ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Camada de saída (sem ativação, pois a CrossEntropyLoss aplica Softmax)
        x = self.fc3(x)

        return x