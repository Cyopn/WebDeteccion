# Sistema de Detección de Personas

Sistema de detección y seguimiento de personas en tiempo real utilizando YOLOv11 y OpenCV, con interfaz web mediante Flask.


## Requisitos del Sistema

### Software Base
- Python 3.8 o superior
- FFmpeg (para procesamiento de video)
- pip (gestor de paquetes de Python)

## Instalación

### 1. Instalar FFmpeg

1. Descargar desde [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extraer y agregar al PATH del sistema

#### Verificar instalación:
```bash
ffmpeg -version
```

### 3. Crear Entorno Virtual

```bash
python3 -m venv .venv
.venv\Scripts\activate
```

### 4. Instalar Dependencias de Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Dependencias principales:
- **Flask** (>=2.0.0): Framework web
- **opencv-python-headless** (>=4.5.5.64): Procesamiento de visión por computadora
- **numpy** (>=1.21): Operaciones numéricas
- **ultralytics** (>=8.0.0): YOLOv11 para detección de objetos
- **flask-cors** (>=3.0.10): Manejo de CORS
- **gunicorn** (>=20.1.0): Servidor WSGI para producción

### 5. Descargar Modelo YOLOv11 (si no está incluido)

El archivo `yolo11n.pt` debería estar en el repositorio. Si no está presente:

```bash
# El modelo se descargará automáticamente al ejecutar la aplicación por primera vez
# O descargarlo manualmente desde Python:
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt')"
```

## Uso

### Opción 1: Ejecutar con menú interactivo (RECOMENDADO)

```bash
source .venv/bin/activate
python run.py
```

Esto te mostrará un menú para elegir:
- **GPU (CUDA)**: Para computadoras con GPU NVIDIA (5-10x más rápido)
- **CPU**: Para cualquier computadora
- **Auto**: Detecta automáticamente si tienes GPU disponible

### Opción 2: Cambiar dispositivo manualmente

```bash
source .venv/bin/activate

# Ver configuración actual
python device_config.py info

# Cambiar a GPU
python device_config.py set cuda

# Cambiar a CPU
python device_config.py set cpu

# Cambiar a Auto
python device_config.py set auto

# Ejecutar aplicación
python app.py
```

## Selección de Dispositivo (CPU/GPU)

### Archivos de configuración:
- **device_config.py**: Gestiona la configuración de dispositivo
- **device_config.json**: Archivo de configuración (creado automáticamente)

### Cambiar dispositivo en tiempo de ejecución:

```bash
# Ver configuración actual
python device_config.py info

# Cambiar a GPU
python device_config.py set cuda

# Cambiar a CPU
python device_config.py set cpu

# Cambiar a Auto (detecta automáticamente)
python device_config.py set auto

# Restaurar valores por defecto
python device_config.py reset
```

### Configuración automática:
- Si configuras en **auto**, la aplicación detectará automáticamente:
  - Si tienes GPU NVIDIA disponible, usa GPU
  - Si no, usa CPU


## Características del Sistema de Detección

### CentroidTracker
- **Seguimiento persistente**: Asigna IDs únicos a cada persona
- **Manejo de oclusiones**: Mantiene el seguimiento cuando personas desaparecen temporalmente
- **Colores distintivos**: Cada persona tiene un color único para fácil identificación

## Solución de Problemas

### Error: "No module named 'cv2'"
```bash
pip install opencv-python-headless
```

### Error: "FFmpeg not found"
Instalar FFmpeg según las instrucciones de tu sistema operativo (ver sección de instalación).

### Error de memoria con videos grandes
Reducir el tamaño del frame o la tasa de procesamiento en `app.py`:
```python
# Redimensionar frame
frame = cv2.resize(frame, (640, 480))
```

## Rendimiento

### Requisitos Recomendados
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **GPU**: NVIDIA (opcional, mejora significativamente el rendimiento)

### Aceleración GPU con CUDA

#### Verificar GPU disponible:
```bash
nvidia-smi
```

#### Verificar si CUDA está funcionando:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible')"
```

#### Archivos de configuración CUDA:
- `cuda_config.py`: Configuración automática de GPU
- Se ejecuta automáticamente al iniciar `app.py`

#### Rendimiento esperado:
- **CPU**: ~2-5 FPS (videos)
- **GPU RTX 3050**: ~15-30 FPS (videos)
- **Mejora**: 5-10x más rápido con GPU

### Optimización de memoria GPU
Para RTX 3050 (4GB VRAM):
```bash
# Monitorear GPU mientras corre:
watch -n 1 nvidia-smi
```

Si hay errores de memoria:
```bash
# Reducir tamaño de frame en app.py
frame = cv2.resize(frame, (640, 480))

# O usar modelo más pequeño (YOLOv11 nano es el más ligero)
```