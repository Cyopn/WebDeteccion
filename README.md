# Sistema de Detección de Objetos

Sistema de detección y seguimiento de objetos en tiempo real utilizando modelos preentrenados y OpenCV, con interfaz web mediante Flask.

## Funciones Principales

- **Detección en tiempo real**: Detecta objetos en videos e imágenes usando el modelo SSD MobileNet COCO.
- **Seguimiento de objetos**: Asigna IDs únicos y mantiene el seguimiento persistente en la detección en videos.
- **Interfaz web**: Acceso vía navegador para subir imágenes/videos y ver resultados.
- **Aceleración GPU/CPU**: Configuración automática o manual para optimizar rendimiento.
- **Procesamiento de video**: Soporte para archivos de video con FFmpeg.

## Instalación

### Requisitos
- Python 3.8+
- FFmpeg (descargar de [ffmpeg.org](https://ffmpeg.org/download.html) y agregar al PATH)

### Pasos
1. Crear entorno virtual: `python -m venv .venv`
2. Activar: `.venv\Scripts\activate` (Windows) o `source .venv/bin/activate` (Linux/Mac)
3. Instalar dependencias: `pip install -r requirements.txt`
4. Ejecutar: `python run.py` (menú interactivo) o `python app.py`

Para más detalles, ver la sección completa de instalación abajo.

## Client

`client/` contiene la interfaz web del usuario para interactuar con el sistema de detección. Incluye:

- **index.html**: Página principal con formulario para subir imágenes/videos y mostrar resultados de detección.
- **main.js**: Lógica JavaScript para manejar la comunicación con el servidor Flask, procesar archivos y mostrar detecciones en tiempo real.
- **style.css**: Estilos CSS para la interfaz de usuario, asegurando una experiencia visual atractiva y responsiva.

Esta interfaz permite a los usuarios subir archivos multimedia y recibir resultados de detección sin necesidad de comandos de terminal.

## Documentación

Los archivos de documentación proporcionan una descripción detallada línea por línea del código fuente:

- **generate_doc_app.py**: Script Python que analiza `app.py` y genera `doc_app.md`, un archivo con descripciones de cada línea del código del servidor Flask.
- **generate_doc_client.py**: Script Python que analiza los archivos en `client/` (como `main.js` e `index.html`) y genera `doc_client.md`, con explicaciones línea por línea del código del cliente web.
- **doc_app.md**: Archivo generado con documentación del backend (app.py).
- **doc_client.md**: Archivo generado con documentación del frontend (client/).


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
- **device_config.json**: Archivo de configuración

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
