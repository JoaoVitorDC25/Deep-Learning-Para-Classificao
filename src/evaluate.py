import torch

def evaluate_model(loader_teste, device, model,classes):
    
    # Desativa o cálculo de gradientes (não é necessário durante a avaliação)
    with torch.no_grad():

        # Inicializa contador de acertos totais
        n_correct = 0

        # Inicializa contador total de amostras
        n_samples = 0

        # Lista para contar acertos por classe
        n_class_correct = [0 for _ in range(10)]

        # Lista para contar total de amostras por classe
        n_class_samples = [0 for _ in range(10)]

        # Loop sobre o conjunto de teste
        for images, labels in loader_teste:

            # Move imagens para o dispositivo (CPU ou GPU)
            images = images.to(device)

            # Move rótulos para o mesmo dispositivo
            labels = labels.to(device)

            # Faz a inferência com o modelo
            outputs = model(images)

            # Obtém as classes com maior probabilidade
            _, predicted = torch.max(outputs, 1)

            # Incrementa o número total de amostras
            n_samples += labels.size(0)

            # Incrementa o número total de acertos
            n_correct += (predicted == labels).sum().item()

            # Calcular acurácia por classe
            for i in range(len(labels)):

                # Obtém o rótulo verdadeiro
                label = labels[i]

                # Obtém o rótulo previsto
                pred = predicted[i]

                # Se acertou, incrementa o contador da classe
                if (label == pred):
                    n_class_correct[label] += 1

                # Incrementa o total de amostras da classe
                n_class_samples[label] += 1

        # Calcula a acurácia geral do modelo
        acc_geral = 100.0 * n_correct / n_samples

        # Exibe o resultado da acurácia geral
        print(f'Acurácia geral do modelo na base de teste: {acc_geral:.2f} %')

        # Imprime linha separadora
        print("-" * 30)

        # Loop para calcular e exibir acurácia de cada classe
        for i in range(10):

            # Se houver amostras para a classe
            if n_class_samples[i] > 0:

                # Calcula a acurácia da classe
                acc_classe = 100.0 * n_class_correct[i] / n_class_samples[i]

                # Exibe acurácia da classe
                print(f'Acurácia da classe {classes[i]}: {acc_classe:.2f} %')

            # Caso não existam amostras da classe
            else:

                # Indica ausência de dados para a classe
                print(f'Acurácia da classe {classes[i]}: N/A (sem amostras)\n')