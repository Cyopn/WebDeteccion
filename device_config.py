import os
import json

CONFIG_FILE = 'device_config.json'

DEFAULT_CONFIG = {
    'device': 'auto',
    'device_id': 0,
    'half_precision': False,
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuración guardada en {CONFIG_FILE}")


def get_device():
    config = load_config()
    device_setting = config.get('device', 'auto').lower()

    if device_setting == 'auto':
        # Prefer PyTorch detection, fall back to TensorFlow if PyTorch not available
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except Exception:
            pass

        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                return 'cuda'
        except Exception:
            pass

        return 'cpu'

    return device_setting


def get_device_name():
    device = get_device()

    if device == 'cuda':
        # Try PyTorch first (preferred), then TensorFlow, then best-effort heuristics
        try:
            import torch
            try:
                name = torch.cuda.get_device_name(0)
                return f"GPU: {name}"
            except Exception:
                # torch available but couldn't get name
                return "GPU: (PyTorch detected GPU, name unknown)"
        except Exception:
            pass

        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Try to extract a readable name if possible
                try:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    name = details.get('device_name') or details.get('name')
                    if name:
                        return f"GPU: {name}"
                except Exception:
                    # Last resort: return TensorFlow's device string
                    try:
                        return f"GPU: {gpus[0].name}"
                    except Exception:
                        pass
        except Exception:
            pass

        return "GPU: Unknown"
    else:
        return "CPU"


def print_device_info():
    config = load_config()
    device = get_device()
    device_name = get_device_name()

    print("\n" + "="*50)
    print("Configuración de Dispositivo")
    print("="*50)
    print(f"Modo configurado: {config.get('device', 'auto').upper()}")
    print(f"Dispositivo actual: {device_name}")
    # Additional diagnostics
    try:
        import importlib
        torch_spec = importlib.util.find_spec('torch')
        tf_spec = importlib.util.find_spec('tensorflow')
        print(f"PyTorch installed: {bool(torch_spec)}")
        print(f"TensorFlow installed: {bool(tf_spec)}")
    except Exception:
        pass

    # Try to show nvidia-smi output (if available)
    try:
        import subprocess
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], stderr=subprocess.STDOUT, universal_newlines=True)
        print("nvidia-smi:")
        print(out.strip())
    except Exception:
        pass

    print(f"Half precision: {config.get('half_precision', False)}")
    print("="*50 + "\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'info':
            print_device_info()

        elif command == 'set':
            if len(sys.argv) < 3:
                print("Uso: python device_config.py set <cuda|cpu|auto>")
                sys.exit(1)

            device = sys.argv[2].lower()
            if device not in ['cuda', 'cpu', 'auto']:
                print("Dispositivo inválido. Usa: cuda, cpu o auto")
                sys.exit(1)

            config = load_config()
            config['device'] = device
            save_config(config)

            print(f"Dispositivo cambiado a: {device.upper()}")
            print_device_info()

        elif command == 'half':
            if len(sys.argv) < 3:
                print("Uso: python device_config.py half <true|false>")
                sys.exit(1)

            half = sys.argv[2].lower() == 'true'
            config = load_config()
            config['half_precision'] = half
            save_config(config)

            status = "ACTIVADA" if half else "DESACTIVADA"
            print(f"Precisión media: {status}")
            print_device_info()

        elif command == 'reset':
            save_config(DEFAULT_CONFIG.copy())
            print("Configuración restaurada a valores por defecto")
            print_device_info()

        else:
            print("Comandos disponibles:")
            print(
                "  python device_config.py info          - Mostrar configuración actual")
            print(
                "  python device_config.py set <tipo>    - Cambiar dispositivo (cuda|cpu|auto)")
            print(
                "  python device_config.py half <true|false> - Activar/desactivar precisión media")
            print(
                "  python device_config.py reset         - Restaurar valores por defecto")
    else:
        print_device_info()
