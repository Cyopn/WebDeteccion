# Guía Rápida: Selección de Dispositivo (CPU/GPU)

Este proyecto detecta automáticamente si tienes GPU disponible y puede usar CPU o GPU para procesar las detecciones. Esto permite que cualquiera pueda usarlo, con o sin GPU NVIDIA.

## Inicio Rápido (3 pasos)

### Paso 1: Abrir terminal y activar entorno virtual

```bash
cd /ruta/al/proyecto
source .venv/bin/activate
```

### Paso 2: Ejecutar con menú interactivo (RECOMENDADO)

```bash
python run.py
```

### Paso 3: Seleccionar opción

```
1) GPU (CUDA) - Más rápido, requiere NVIDIA
2) CPU - Compatible con cualquier computadora
3) Auto - Detecta automáticamente
4) Configuración avanzada
5) Salir
```

## Casos de Uso

### GPU NVIDIA RTX/GTX

1. Ejecuta: `python run.py`
2. Selecciona opción **1 (GPU)**

### Sin GPU

1. Ejecuta: `python run.py`
2. Selecciona opción **2 (CPU)**

### No sé si tengo GPU compatible

1. Ejecuta: `python run.py`
2. Selecciona opción **3 (Auto)**
3. El sistema detectará automáticamente

## Cambiar dispositivo sin menú

```bash
# Ver configuración actual
python device_config.py info

# Cambiar a GPU
python device_config.py set cuda

# Cambiar a CPU
python device_config.py set cpu

# Cambiar a Auto
python device_config.py set auto

# Restaurar valores por defecto
python device_config.py reset
```

## Configuración Avanzada

### Half Precision (Precisión Media)

Disponible en opción 4 del menú o mediante:

```bash
# Activar half precision (más rápido, menos preciso)
python device_config.py half true

# Desactivar half precision (más lento, más preciso)
python device_config.py half false
```

**Recomendaciones:**
- **GPU RTX 3050+**: Activar para 10-20% más velocidad
- **GPU antigua (GTX 1060)**: Dejar desactivado
- **CPU**: No tiene efecto, dejar desactivado

## Cambiar dispositivo sin menú

```bash
# Ver configuración actual
python device_config.py info

# Cambiar a GPU
python device_config.py set cuda

# Cambiar a CPU
python device_config.py set cpu

# Cambiar a Auto
python device_config.py set auto

# Restaurar valores por defecto
python device_config.py reset
```

## Archivos importantes

- **`run.py`**: Menú interactivo para seleccionar dispositivo
- **`run.sh`**: Script bash alternativo (Linux/macOS)
- **`device_config.py`**: Gestor de configuración de dispositivo
- **`device_config.json`**: Archivo de configuración (se crea automáticamente)
- **`cuda_config.py`**: Configuración específica de CUDA/GPU

## Rendimiento esperado

| Dispositivo | FPS (Video) | Características |
|-------------|------------|-----------------|
| CPU (i5/i7) | 2-5 | Compatible, consume más energía |
| GPU RTX 3050 | 15-30 | 5-10x más rápido, menos energía |
