#!/usr/bin/env python
import os
import sys
import platform
from device_config import load_config, save_config, print_device_info, get_device


def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def show_menu():
    clear_screen()
    print("=" * 50)
    print("WebDeteccion - Detector de Personas")
    print("=" * 50)
    print()

    print_device_info()

    print("Selecciona cómo ejecutar:")
    print("  1) GPU (CUDA) - Más rápido, requiere NVIDIA")
    print("  2) CPU - Compatible con cualquier computadora")
    print("  3) Auto - Detectar automáticamente")
    print("  4) Configuración avanzada")
    print("  5) Salir")
    print()


def get_choice():
    while True:
        try:
            choice = input("Opción (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("Opción no válida. Por favor, ingresa 1-5.")
        except KeyboardInterrupt:
            print("\nSaliendo...")
            sys.exit(0)


def set_device(device, pause=True):
    config = load_config()
    config['device'] = device
    save_config(config)
    print(f"Dispositivo cambiado a: {device.upper()}")
    if pause:
        input("Presiona Enter para continuar...")


def set_half_precision():
    clear_screen()
    print("=" * 50)
    print("WebDeteccion - Configuración de Precisión")
    print("=" * 50)
    print()

    config = load_config()
    current = "ACTIVADA" if config.get('half_precision') else "DESACTIVADA"
    print(f"Precisión Media (float16): {current}")
    print()
    print("⚠ Advertencia:")
    print("  - Activar: Más rápido, menos memoria GPU, puede afectar precisión")
    print("  - Desactivar: Más lento, más memoria, máxima precisión")
    print()

    while True:
        choice = input("Deseas activar? (s/n): ").strip().lower()
        if choice in ['s', 'n', 'si', 'no', 'y', 'yes']:
            break
        print("Opción no válida.")

    half = choice in ['s', 'si', 'y', 'yes']
    config['half_precision'] = half
    save_config(config)

    status = "ACTIVADA" if half else "DESACTIVADA"
    print(f"\nPrecisión media: {status}")
    input("Presiona Enter para continuar...")


def show_advanced_menu():
    while True:
        clear_screen()
        print("=" * 50)
        print("WebDeteccion - Configuración Avanzada")
        print("=" * 50)
        print()
        print_device_info()
        print()
        print("Opciones:")
        print("  1) Cambiar precisión (half/full)")
        print("  2) Volver al menú principal")
        print()

        choice = input("Opción (1-2): ").strip()
        if choice == '1':
            set_half_precision()
        elif choice == '2':
            break
        else:
            print("Opción no válida.")
            input("Presiona Enter...")


def run_app():
    print("\nIniciando aplicación...")
    print("La aplicación está disponible en: http://localhost:5000")
    print("Presiona CTRL+C para detener")
    print()

    os.system('python app.py')


def main():
    while True:
        show_menu()
        choice = get_choice()

        if choice == '1':
            set_device('cuda', pause=False)
            clear_screen()
            run_app()
            break
        elif choice == '2':
            set_device('cpu', pause=False)
            clear_screen()
            run_app()
            break
        elif choice == '3':
            set_device('auto', pause=False)
            clear_screen()
            run_app()
            break
        elif choice == '4':
            show_advanced_menu()
        elif choice == '5':
            print("Saliendo...")
            sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAplicación interrumpida.")
        sys.exit(0)
