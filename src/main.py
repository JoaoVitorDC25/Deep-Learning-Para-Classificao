import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from select_device import select_device
from visualizations import imshow
from convNet import ConvNet
from deploy import classifies_image
from train import train_model
from evaluate import evaluate_model
import config as cfg


# Seleção de dispositivo (CUDA, MPS ou CPU)
device = select_device()

# Hiperparâmetros do modelo
num_epochs = cfg.NUM_EPOCHS        # Número de épocas para treinar
batch_size = cfg.BATCH_SIZE         # Tamanho do lote (batch)
learning_rate = cfg.LEARNING_RATE   # Taxa de aprendizado

# Definir as transformações para os dados
dsa_transformacoes = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Baixar e carregar o dataset de treino
dataset_treino = torchvision.datasets.CIFAR10(root = '../data',
                                                  train = True,
                                                  download = True,
                                                  transform = dsa_transformacoes)

# Baixar e carregar o dataset de teste
dataset_teste = torchvision.datasets.CIFAR10(root = '../data',
                                                 train = False,
                                                 download = True,
                                                 transform = dsa_transformacoes)

# Criar os DataLoaders para carregar os dados em lotes
loader_treino = torch.utils.data.DataLoader(dataset_treino,
                                                batch_size = batch_size,
                                                shuffle = True)

loader_teste = torch.utils.data.DataLoader(dataset_teste,
                                               batch_size = batch_size,
                                               shuffle = False)

# Definir as classes do CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Número de imagens de treino: {len(dataset_treino)}")
print(f"Número de imagens de teste: {len(dataset_teste)}")
print(f"Número de lotes (batches) de treino: {len(loader_treino)}")
print(f"Número de lotes (batches) de teste: {len(loader_teste)}")

# Obtém um lote de imagens de treino
dataiter = iter(loader_treino)
images, labels = next(dataiter)

# Mostra as primeiras 4 imagens do lote
print("Amostra de imagens de treino:")
imshow(torchvision.utils.make_grid(images[:4]))

# Imprime os labels correspondentes
print('Labels: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Vamos instanciar o modelo e movê-lo para a CPU para visualizarmos um resumo do modelo
modelo_dsa = ConvNet().to("cpu")

print("Arquitetura do Modelo:")
print(modelo_dsa)

# Sumário
summary(modelo_dsa, (3, 32, 32), device = "cpu")

# Agora instanciamos o modelo e movemos para o dispositivo de treino (GPU/CPU)
modelo_dsa = ConvNet().to(device)
print(device)

# Otimizador Adam
optimizer = optim.Adam(modelo_dsa.parameters(), lr = learning_rate)

train_model(loader_treino, loader_teste, modelo_dsa, optimizer, device)

# Coloca o modelo em modo de avaliação
modelo_dsa.eval()
evaluate_model(loader_teste, device, modelo_dsa, classes)

PATH = '../model/model.pth'
torch.save(modelo_dsa.state_dict(), PATH)
print(f'Modelo salvo em: {PATH}')

# Cria uma nova instância do modelo
model_carregado = ConvNet().to(device)

# Carrega os pesos (state_dict) salvos
model_carregado.load_state_dict(torch.load(PATH))

# IMPORTANTE: Colocar o modelo em modo de avaliação
model_carregado.eval()

# Definir a transformação para imagens de inferência
# Deve ser a MESMA transformação usada no treino/teste
# (Exceto por augmentations, que não usamos aqui)
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)), # Garantir que a imagem tenha 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem1.jpg",
    model_carregado,
    inference_transform,
    device,
    classes
)

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem2.jpg",
    model_carregado,
    inference_transform,
    device,
    classes
)

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem3.png",
    model_carregado,
    inference_transform,
    device,
    classes
)

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem4.jpg",
    model_carregado,
    inference_transform,
    device,
    classes
)

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem5.jpg",
    model_carregado,
    inference_transform,
    device,
    classes
)

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem6.jpg",
    model_carregado,
    inference_transform,
    device,
    classes
)

# Usamos o modelo para classificar a imagem
classifies_image(
    "../images/imagem7.jpeg",
    model_carregado,
    inference_transform,
    device,
    classes
)