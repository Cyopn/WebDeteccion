import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA no está disponible. El proyecto funcionará en CPU.")
        return False

    print("CUDA disponible")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN: {torch.backends.cudnn.version()}")
    print(
        f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print(f"  cuDNN enabled: True")
    print(f"  cuDNN benchmark: True")

    return True


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def memory_info():
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n  Memoria GPU:")
    print(f"    Asignada: {allocated:.2f} GB")
    print(f"    Reservada: {reserved:.2f} GB")
    print(f"    Total: {total:.2f} GB")
    print(f"    Disponible: {total - allocated:.2f} GB")


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print("Configuración CUDA RTX 3050 Mobile\n" + "="*40)
    check_cuda()
    memory_info()
    print("="*40)
