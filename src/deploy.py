import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def classifies_image(image_path, model, inference_transform, device, classes):
    """
    Carrega uma imagem local, aplica as transformações
    e faz a predição usando o modelo treinado.
    """

    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erro ao carregar imagem local {image_path}: {e}")
        return

    # Aplica as transformações recebidas como parâmetro
    img_tensor = inference_transform(img_pil)

    # Adiciona a dimensão do lote e move para o dispositivo
    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_idx = probabilities.max(dim=1)

    classe_predita = classes[predicted_idx.item()]
    confianca = confidence.item() * 100

    plt.imshow(img_pil)
    plt.title(
        f"Classe prevista: {classe_predita} "
        f"(Confiança: {confianca:.2f}%)"
    )
    plt.axis("off")
    plt.show()

    return classe_predita, confianca