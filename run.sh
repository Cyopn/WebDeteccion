#!/bin/bash

echo "======================================"
echo "WebDeteccion - Detector de Personas"
echo "======================================"
echo ""

python device_config.py info

echo "Selecciona cómo ejecutar:"
echo "  1) GPU (CUDA) - Más rápido"
echo "  2) CPU - Compatible con cualquier computadora"
echo "  3) Auto - Detectar automáticamente"
echo "  4) Salir"
echo ""
read -p "Opción (1-4): " option

case $option in
    1)
        echo "Cambiando a GPU..."
        python device_config.py set cuda
        ;;
    2)
        echo "Cambiando a CPU..."
        python device_config.py set cpu
        ;;
    3)
        echo "Cambiando a Auto..."
        python device_config.py set auto
        ;;
    4)
        echo "Saliendo..."
        exit 0
        ;;
    *)
        echo "Opción no válida. Usando configuración actual."
        ;;
esac

echo ""
echo "Iniciando aplicación..."
echo ""

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python app.py
