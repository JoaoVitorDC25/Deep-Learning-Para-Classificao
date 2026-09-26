import torch


def select_device():
    if torch.cuda.is_available():
        # Prioridade 1: GPU NVIDIA (CUDA)
        device = torch.device("cuda")
        print("Dispositivo selecionado: GPU NVIDIA (CUDA)")

    elif torch.backends.mps.is_available():
        # Prioridade 2: GPU Apple (MPS)
        device = torch.device("mps")
        print("Dispositivo selecionado: GPU Apple (MPS)")

    else:
        # Fallback: CPU
        device = torch.device("cpu")
        print("Dispositivo selecionado: CPU")

    print(f'Usando dispositivo: {device}')
    
    return device