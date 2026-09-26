
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from torchsummary import summary


import torch
import config as cfg



def train_model(loader_treino, loader_teste, model, optimizer, device):
    """
    Função para treinar o modelo de rede neural convolucional (ConvNet).
    """
    # Define a função de perda. O modelo vai buscar os parâmetros que reduzem o erro geral das previsões.
    criterion = nn.CrossEntropyLoss()
    
    print("\nIniciando o treinamento...\n")

    # Calcula o total de passos por época (número de batches)
    n_total_steps = len(loader_treino)

    # Loop principal de treinamento
    for epoch in range(cfg.NUM_EPOCHS):

        # Coloca o modelo em modo de treinamento
        model.train()

        # Inicializa o acumulador de perda
        running_loss = 0.0

        # Itera sobre os batches do conjunto de treino
        for i, (images, labels) in enumerate(loader_treino):

        # Move os tensores (imagens e rótulos) para o dispositivo (CPU ou GPU), próximos do modelo
            images = images.to(device)
            labels = labels.to(device)

        # Passagem para frente (forward)
        # Aqui ocorre a previsão do modelo
            outputs = model(images)

        # Calcula o erro do modelo
            loss = criterion(outputs, labels)

        # Zera os gradientes acumulados de iterações anteriores
            optimizer.zero_grad()

        # Calcula os gradientes via backpropagation
            loss.backward()

        # Atualiza os pesos do modelo
            optimizer.step()

        # Soma o valor da perda para cálculo médio posterior
            running_loss += loss.item()

    # Após cada época, avalia o modelo no conjunto de teste (validação)
    # Coloca o modelo em modo de avaliação
        model.eval()

    # Desativa o cálculo de gradientes para economizar memória e tempo
        with torch.no_grad():

            n_correct = 0   # Contador de acertos
            n_samples = 0   # Contador de amostras

        # Loop sobre o conjunto de teste
            for val_images, val_labels in loader_teste:

            # Move imagens e rótulos para o dispositivo
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

            # Faz a inferência no conjunto de teste
                val_outputs = model(val_images)

            # torch.max retorna (valor, índice) → pegamos o índice da classe prevista
                _, predicted = torch.max(val_outputs.data, 1)

            # Incrementa o total de amostras
                n_samples += val_labels.size(0)

            # Incrementa o número de acertos
                n_correct += (predicted == val_labels).sum().item()

    # Calcula a acurácia e a perda média da época
        acc = 100.0 * n_correct / n_samples
        avg_loss = running_loss / n_total_steps

    # Exibe métricas de desempenho para a época atual
        print(f'Epoch [{epoch+1}/{cfg.NUM_EPOCHS}], Erro em Treino: {avg_loss:.4f}, Acurácia em Teste: {acc:.2f} %')

# Exibe mensagem final de conclusão
    print('\nTreinamento finalizado.\n')