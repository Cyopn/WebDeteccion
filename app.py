from flask import Flask, request, jsonify, send_file, Response, render_template, send_from_directory
import numpy as np
import cv2
import io
import base64
import tempfile
import os
from flask_cors import CORS
import threading
import subprocess
import time
import hashlib
import io
import logging
import sys
from collections import deque
import shutil

# Reducir verbosidad de TensorFlow y suprimir warnings deprecados
import warnings
# 0=all,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except Exception:
    pass

try:
    from device_config import get_device, print_device_info, load_config
    print_device_info()
    DEVICE = get_device()
except ImportError:
    print("device_config no disponible, usando CPU")
    DEVICE = 'cpu'

if DEVICE == 'cuda':
    try:
        from cuda_config import check_cuda, empty_cache
        check_cuda()
    except ImportError:
        print("cuda_config no disponible, usando CPU")
        DEVICE = 'cpu'

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
except Exception as e:
    TF_AVAILABLE = False
    TF_VERSION = None
    print(f"Error cargando TensorFlow: {e}")


class CentroidTracker:

    def __init__(self, maxDisappeared=30, maxDistance=100):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.object_colors = {}

    def register(self, centroid):
        color = self._generate_color(self.nextObjectID)
        self.object_colors[self.nextObjectID] = color
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.object_colors[objectID]

    def _generate_color(self, object_id):
        hue = (object_id * 60) % 180
        saturation = 200 + (object_id * 30) % 55
        value = 200 + (object_id * 20) % 55
        bgr = cv2.cvtColor(
            np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)
        return tuple(int(x) for x in bgr[0][0])

    def update(self, detections):
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return detections

        input_centroids = np.zeros((len(detections), 2))
        for i, det in enumerate(detections):
            x = det.get('x', 0) + det.get('w', 0) / 2
            y = det.get('y', 0) + det.get('h', 0) / 2
            input_centroids[i] = [x, y]

        if len(self.objects) > 0:
            objectIDs = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            D = np.zeros((len(objectIDs), len(input_centroids)))
            for i, object_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    dist = np.linalg.norm(object_centroid - input_centroid)
                    D[i, j] = dist

            matched_indices = []
            for i in range(len(objectIDs)):
                if D[i].size == 0:
                    continue
                j = np.argmin(D[i])
                if D[i, j] < self.maxDistance:
                    if j not in [m[1] for m in matched_indices]:
                        matched_indices.append((i, j))

            unused_objectIDs = [objectIDs[i] for i in range(len(objectIDs))
                                if i not in [m[0] for m in matched_indices]]
            for objectID in unused_objectIDs:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for i, j in matched_indices:
                objectID = objectIDs[i]
                self.objects[objectID] = input_centroids[j]
                self.disappeared[objectID] = 0
                detections[j]['person_id'] = objectID
                detections[j]['person_color'] = self.object_colors[objectID]

            unused_input_indices = [j for j in range(len(input_centroids))
                                    if j not in [m[1] for m in matched_indices]]
            for j in unused_input_indices:
                self.register(input_centroids[j])
                last_id = self.nextObjectID - 1
                detections[j]['person_id'] = last_id
                detections[j]['person_color'] = self.object_colors[last_id]
        else:
            for j, input_centroid in enumerate(input_centroids):
                self.register(input_centroid)
                last_id = self.nextObjectID - 1
                detections[j]['person_id'] = last_id
                detections[j]['person_color'] = self.object_colors[last_id]

        return detections


app = Flask(__name__, static_folder='client',
            static_url_path='', template_folder='client')
CORS(app)
LOG_BUFFER_SIZE = 1000
_log_buffer = deque(maxlen=LOG_BUFFER_SIZE)
_log_lock = threading.Lock()
_created_tmp_files = {}
_created_tmp_lock = threading.Lock()
TMP_FILE_TTL = 60 * 10
TMP_CLEAN_INTERVAL = 60 * 5


def _append_log(line: str):
    try:
        with _log_lock:
            if line is None:
                return
            s = str(line)
            for part in s.splitlines():
                _log_buffer.append(part)
    except Exception:
        pass


def _register_tmp_file(path: str):
    try:
        with _created_tmp_lock:
            _created_tmp_files[path] = time.time()
            _append_log(f"TMP_REGISTER {path}")
    except Exception:
        pass


def _append_response_log(path: str, status: int, **kwargs):
    try:
        extra = ' '.join(f"{k}={v}" for k, v in kwargs.items())
        _append_log(f"RESPONSE path={path} status={int(status)} {extra}")
    except Exception:
        pass


_video_tracker = CentroidTracker(maxDisappeared=30, maxDistance=100)
_stream_tracker = CentroidTracker(maxDisappeared=20, maxDistance=80)
_tracker_lock = threading.Lock()


def _temp_cleanup_worker():
    while True:
        try:
            now = time.time()
            to_remove = []
            with _created_tmp_lock:
                for p, ts in list(_created_tmp_files.items()):
                    try:
                        if now - ts > TMP_FILE_TTL:
                            to_remove.append(p)
                    except Exception:
                        to_remove.append(p)
            for p in to_remove:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                        _append_log(f"TMP_CLEANED {p}")
                except Exception as e:
                    _append_log(f"TMP_CLEAN_ERROR {p}: {e}")
                try:
                    with _created_tmp_lock:
                        _created_tmp_files.pop(p, None)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            time.sleep(TMP_CLEAN_INTERVAL)
        except Exception:
            time.sleep(60)


try:
    t_cleanup = threading.Thread(target=_temp_cleanup_worker, daemon=True)
    t_cleanup.start()
except Exception:
    pass


class BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            _append_log(msg)
        except Exception:
            pass


class StderrWrapper:
    def __init__(self, orig):
        self.orig = orig
        self._buf = ''

    def write(self, data):
        try:
            try:
                self.orig.write(data)
            except Exception:
                pass
            if not data:
                return
            self._buf += str(data)
            if '\n' in self._buf:
                parts = self._buf.split('\n')
                for p in parts[:-1]:
                    if p:
                        _append_log(p)
                self._buf = parts[-1]
        except Exception:
            pass

    def flush(self):
        try:
            if self._buf:
                _append_log(self._buf)
                self._buf = ''
        except Exception:
            pass


try:
    logging.getLogger('werkzeug').setLevel(logging.INFO)
except Exception:
    pass

# Variables para modelo TensorFlow Hub SSD MobileNet
tf_hub_model = None
tf_hub_lock = threading.Lock()
TF_HUB_MOBILENET_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"


def translate_label(label: str) -> str:
    """Traduce etiquetas al español"""
    if label is None:
        return ''
    lbl = str(label).strip().lower()
    mapping = {
        'person': 'persona', 'people': 'persona', 'persona': 'persona',
    }
    return mapping.get(lbl, lbl)


# COCO class names (index matches TF detection_classes which are 1-based for COCO)
COCO_CLASSES = [
    'sin etiqueta', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'glasses', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'hair brush'
]


# Funciones para modelos de TensorFlow Hub
def get_tf_hub_mobilenet_model():
    """Carga el modelo TensorFlow Hub SSD MobileNet con caché"""
    global tf_hub_model
    if not TF_AVAILABLE:
        _append_log("ERROR: TensorFlow no disponible")
        return None

    if tf_hub_model is None:
        with tf_hub_lock:
            if tf_hub_model is None:
                try:
                    _append_log(
                        f"Cargando SSD MobileNet v2 de TensorFlow Hub...")
                    model = hub.load(TF_HUB_MOBILENET_URL)
                    tf_hub_model = model
                    _append_log(f"✓ SSD MobileNet v2 cargado exitosamente")
                except Exception as e:
                    _append_log(f"✗ Error cargando modelo: {e}")
                    return None

    return tf_hub_model


def _nms_boxes(boxes, scores, iou_threshold=0.5):
    # boxes: list of [x1,y1,x2,y2]
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return list(keep)


def _dedupe_detections(detections, iou_threshold=0.45):
    """Elimina detecciones muy solapadas dejando la de mayor confianza."""
    if not detections:
        return detections
    boxes = []
    scores = []
    for d in detections:
        x = float(d.get('x', 0))
        y = float(d.get('y', 0))
        w = float(d.get('w', 0))
        h = float(d.get('h', 0))
        boxes.append([x, y, x + w, y + h])
        scores.append(float(d.get('confidence', 0.0)))
    keep = _nms_boxes(boxes, scores, iou_threshold=iou_threshold)
    kept = [detections[i] for i in keep]
    return kept


def detect_objects(image, conf_threshold=0.5, class_id=-1, nms_iou=0.5):
    """Detecta objetos usando TensorFlow Hub SSD MobileNet v2.
    - `class_id` : si >=0 filtra por esa clase COCO (1-based index as TF returns), -1 acepta todas.
    - Aplica NMS para eliminar detecciones solapadas.
    """
    if not TF_AVAILABLE:
        _append_log("ERROR: TensorFlow no disponible")
        return []

    try:
        model = get_tf_hub_mobilenet_model()
        if model is None:
            return []

        h, w = image.shape[:2]
        inp_size = 300

        # Preparar imagen: convertir BGR->RGB, redimensionar y crear tensor [1,H,W,3]
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            img_rgb = image.copy()
        resized = cv2.resize(img_rgb, (inp_size, inp_size))

        # Some TF Hub detectors accept uint8 images; keep as uint8 to be safe
        input_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Ejecutar modelo (proteger con lock por seguridad en entornos multi-hilo)
        with tf_hub_lock:
            detections = model(input_tensor)

        results = []
        detection_boxes = detections['detection_boxes'].numpy()[0]
        detection_scores = detections['detection_scores'].numpy()[0]
        detection_classes = detections['detection_classes'].numpy()[0]

        raw_boxes = []
        raw_scores = []
        raw_classes = []
        for i, (box, score, cls) in enumerate(zip(detection_boxes, detection_scores, detection_classes)):
            if score < conf_threshold:
                continue
            cls_int = int(cls)
            # class_id argument refers to COCO index as in client (0 means 'sin etiqueta', 1 person, ...)
            if class_id >= 0:
                # if client provided class index 0 (sin etiqueta) treat as no-match
                if cls_int != int(class_id):
                    continue
            # Coordenadas normalizadas a píxeles
            y1, x1, y2, x2 = box
            x1_px = float(x1 * w)
            y1_px = float(y1 * h)
            x2_px = float(x2 * w)
            y2_px = float(y2 * h)
            raw_boxes.append([x1_px, y1_px, x2_px, y2_px])
            raw_scores.append(float(score))
            raw_classes.append(cls_int)

        # Aplicar NMS sobre raw_boxes
        keep_indices = _nms_boxes(raw_boxes, raw_scores, iou_threshold=nms_iou)

        for idx in keep_indices:
            x1_px, y1_px, x2_px, y2_px = raw_boxes[idx]
            x = int(max(0, round(x1_px)))
            y = int(max(0, round(y1_px)))
            bw = int(max(0, round(x2_px - x1_px)))
            bh = int(max(0, round(y2_px - y1_px)))
            cls_int = raw_classes[idx]
            label = COCO_CLASSES[cls_int] if 0 <= cls_int < len(
                COCO_CLASSES) else str(cls_int)
            results.append({
                'x': x,
                'y': y,
                'w': bw,
                'h': bh,
                'confidence': float(raw_scores[idx]),
                'class_id': int(cls_int),
                'label': translate_label(label)
            })

        # Dedupe final sobre cajas ya en píxeles (redundancia de seguridad)
        try:
            results = _dedupe_detections(results, iou_threshold=0.45)
        except Exception:
            pass

        return results

    except Exception as e:
        _append_log(f"Error en detección TensorFlow: {e}")
        import traceback
        _append_log(traceback.format_exc())
        return []


def draw_boxes(image, detections, color=(0, 255, 0), thickness=2):
    img = image.copy()
    for d in detections:
        x, y, w, h = int(d.get("x", 0)), int(d.get("y", 0)), int(
            d.get("w", 0)), int(d.get("h", 0))
        box_color = color
        if 'person_color' in d:
            try:
                pc = d['person_color']
                if isinstance(pc, (list, tuple)) and len(pc) == 3:
                    box_color = tuple(int(c) % 256 for c in pc)
                else:
                    box_color = color
            except Exception:
                box_color = color
        elif 'class_id' in d:
            cid = int(d.get('class_id', 0))
            box_color = (int((cid * 37) % 256), int((cid * 97) %
                         256), int((cid * 61) % 256))
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness)
        label = d.get('label') or d.get('class', '') or ''
        if 'person_id' in d:
            label = f"#{d['person_id']} {label}"
        conf = d.get('confidence', None)
        if conf is not None:
            try:
                conf_text = f"{conf:.2f}"
            except Exception:
                conf_text = str(conf)
            text = f"{label}: {conf_text}"
        else:
            text = f"{label}"
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(1.0, img.shape[1] / 1000.0))
            txt_th = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, txt_th)
            text_x = x
            text_y = y - 5
            rect_top_left = (text_x, text_y - th - 4)
            rect_bottom_right = (text_x + tw + 4, text_y)
            rect_top_left = (
                max(rect_top_left[0], 0), max(rect_top_left[1], 0))
            rect_bottom_right = (min(rect_bottom_right[0], img.shape[1]), min(
                rect_bottom_right[1], img.shape[0]))
            cv2.rectangle(img, rect_top_left, rect_bottom_right, box_color, -1)
            cv2.putText(img, text, (rect_top_left[0] + 2, rect_bottom_right[1] - 2),
                        font, font_scale, (255, 255, 255), txt_th, cv2.LINE_AA)
    return img


def open_video_capture(source):
    backends = []
    try:
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    except Exception:
        backends = [cv2.CAP_ANY]

    for b in backends:
        try:
            cap_try = cv2.VideoCapture(source, b)
        except Exception:
            try:
                cap_try = cv2.VideoCapture(source)
            except Exception:
                cap_try = None
        if cap_try is not None and cap_try.isOpened():
            return cap_try
        try:
            if cap_try is not None:
                cap_try.release()
        except Exception:
            pass
    return None


@app.route("/", methods=["GET"])
def index():
    """Servir la página principal del cliente"""
    return send_from_directory('client', 'index.html')


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    """Servir archivos estáticos (CSS, JS)"""
    return send_from_directory('client', filename)


@app.route('/detect/image', methods=['POST'])
def detect_image():
    start_time = time.time()
    visualize = request.args.get(
        'visualize', 'false').lower() in ('1', 'true', 'yes')
    img = None
    if 'image' in request.files:
        file = request.files['image']
        data = file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        j = request.get_json(silent=True)
        if j and 'image_b64' in j:
            data = base64.b64decode(j['image_b64'])
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'No image provided. Send multipart form `image` or JSON `image_b64`.'}), 400
    conf = float(request.args.get('conf', 0.4))
    class_id = int(request.args.get('class_id', -1))

    # Detectar objetos usando TensorFlow SSD MobileNet
    if not TF_AVAILABLE:
        return jsonify({'error': 'TensorFlow not available. Run: pip install tensorflow tensorflow-hub'}), 500

    detections = detect_objects(img, conf_threshold=conf, class_id=class_id)

    if visualize:
        out_img = draw_boxes(img, detections)
        _, png = cv2.imencode('.png', out_img)
        elapsed_time = time.time() - start_time
        try:
            _append_response_log('/detect/image', 200,
                                 count=len(detections), model='tf_mobilenet', time_sec=f"{elapsed_time:.2f}")
        except Exception:
            pass
        return Response(png.tobytes(), mimetype='image/png')
    elapsed_time = time.time() - start_time
    try:
        _append_response_log('/detect/image', 200,
                             count=len(detections), model='tf_mobilenet', time_sec=f"{elapsed_time:.2f}")
    except Exception:
        pass
    return jsonify({'detections': detections, 'count': len(detections), 'elapsed_seconds': elapsed_time})


@app.route('/detect/video', methods=['POST'])
def detect_video():
    start_time = time.time()
    frame_step = int(request.form.get(
        'frame_step', request.args.get('frame_step', 1)))
    visualize = request.form.get('visualize', request.args.get(
        'visualize', 'false')).lower() in ('1', 'true', 'yes')
    timeline = request.form.get('timeline', request.args.get(
        'timeline', 'false')).lower() in ('1', 'true', 'yes')
    conf = float(request.form.get('conf', request.args.get('conf', 0.4)))
    class_id = int(request.form.get(
        'class_id', request.args.get('class_id', -1)))
    cap = None
    tmp_in = None
    tmp_out = None
    tmp_mp4 = None
    out_writer = None
    transcode = request.form.get(
        'transcode', request.args.get('transcode', '0'))
    transcode = str(transcode).lower() in ('1', 'true', 'yes')
    try:
        if 'video' in request.files:
            file = request.files['video']
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file.save(tmp_in.name)
            cap = open_video_capture(tmp_in.name)
        else:
            camera_url = request.form.get(
                'camera_url') or request.args.get('camera_url')
            if not camera_url:
                j = request.get_json(silent=True) or {}
                camera_url = j.get('camera_url')
            if not camera_url:
                return jsonify({'error': 'No video provided. Send multipart `video` file or `camera_url`.'}), 400
            cap = open_video_capture(camera_url)
        if cap is None or not cap.isOpened():
            return jsonify({'error': 'No se pudo abrir la fuente de video.'}), 500
        global _video_tracker, _tracker_lock
        with _tracker_lock:
            _video_tracker = CentroidTracker(
                maxDisappeared=30, maxDistance=100)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if fps <= 0 or fps > 120:
            fps = 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        if width <= 0 or width > 4096:
            width = 640
        if height <= 0 or height > 4096:
            height = 480
        if visualize:
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            tmp_out.close()
            try:
                _register_tmp_file(tmp_out.name)
            except Exception:
                pass
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_writer = cv2.VideoWriter(
                tmp_out.name, fourcc, fps, (width, height))
            if not out_writer.isOpened():
                _append_log(
                    f"WARNING: VideoWriter failed to open. fps={fps}, size=({width}, {height})")
                out_writer = None
        frame_idx = 0
        total_detections = 0
        per_frame = []
        max_frames = int(request.form.get(
            'max_frames', request.args.get('max_frames', 0)))
        while True:
            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None
            if not ret or frame is None:
                if tmp_in is not None:
                    break
                retry = 0
                max_retries = 3
                while retry < max_retries and (not ret or frame is None):
                    try:
                        ret, frame = cap.read()
                    except Exception:
                        ret, frame = False, None
                    retry += 1
                if not ret or frame is None:
                    break
            if frame_idx % frame_step == 0:
                # Detectar objetos usando TensorFlow SSD MobileNet
                dets = detect_objects(
                    frame, conf_threshold=conf, class_id=class_id)

                with _tracker_lock:
                    dets = _video_tracker.update(dets)

                total_detections += len(dets)
                per_frame.append(
                    {'frame': frame_idx, 'count': len(dets), 'detections': dets})
                if visualize:
                    out_frame = draw_boxes(frame, dets)
                    try:
                        out_writer.write(out_frame)
                    except Exception:
                        pass
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
        cap.release()
        if out_writer:
            out_writer.release()
        elapsed_time = time.time() - start_time
        result = {'frames_processed': frame_idx,
                  'total_detections': total_detections, 'sample': per_frame[:20], 'fps': fps, 'elapsed_seconds': elapsed_time}

        video_to_send = None
        if timeline:
            if visualize and tmp_out and os.path.exists(tmp_out.name):
                video_to_send = tmp_out.name
            elif tmp_in and os.path.exists(tmp_in.name):
                video_to_send = tmp_in.name
        elif visualize and tmp_out and os.path.exists(tmp_out.name):
            video_to_send = tmp_out.name

        if timeline and video_to_send:
            detection_segments = []
            for pf in per_frame:
                if pf['count'] > 0:
                    frame_num = pf['frame']
                    timestamp = frame_num / fps
                    detection_segments.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'count': pf['count'],
                        'detections': pf['detections']
                    })
            metadata = {
                'fps': fps,
                'total_frames': frame_idx,
                'duration': frame_idx / fps,
                'detections': detection_segments,
                'total_detections': total_detections
            }
            try:
                _append_response_log('/detect/video', 200, frames=frame_idx,
                                     total=total_detections, model='tf_mobilenet', mode='timeline', visualize=visualize)
            except Exception:
                pass

            video_file_to_embed = video_to_send
            _append_log(
                f"Timeline: video_to_send={video_to_send}, visualize={visualize}, file_exists={os.path.exists(video_to_send) if video_to_send else False}")

            if visualize and video_to_send and video_to_send.endswith('.avi') and shutil.which('ffmpeg'):
                if os.path.exists(video_to_send) and os.path.getsize(video_to_send) > 1000:
                    _append_log(
                        f"Timeline+Visualize: transcoding AVI to MP4 for embedding (size: {os.path.getsize(video_to_send)} bytes)")
                    try:
                        tmp_mp4_embed = tempfile.NamedTemporaryFile(
                            delete=False, suffix='.mp4')
                        tmp_mp4_embed.close()
                        try:
                            _register_tmp_file(tmp_mp4_embed.name)
                        except Exception:
                            pass
                        cmd = [
                            'ffmpeg', '-y', '-i', video_to_send,
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p',
                            '-movflags', 'faststart', tmp_mp4_embed.name
                        ]
                        proc = subprocess.run(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                        if proc.returncode == 0 and os.path.exists(tmp_mp4_embed.name) and os.path.getsize(tmp_mp4_embed.name) > 1000:
                            video_file_to_embed = tmp_mp4_embed.name
                            _append_log(
                                f"Transcode success: {tmp_mp4_embed.name} (size: {os.path.getsize(tmp_mp4_embed.name)} bytes)")
                        else:
                            _append_log(
                                f"Transcode failed/empty: returncode={proc.returncode}")
                            if proc.stderr:
                                _append_log(
                                    f"FFmpeg stderr: {proc.stderr[:200]}")
                    except Exception as e:
                        _append_log(f"Transcode exception: {e}")
                else:
                    _append_log(
                        f"Video file too small or missing for transcode: {video_to_send}")

            try:
                import base64
                _append_log(
                    f"Attempting to read video file: {video_file_to_embed} (exists: {os.path.exists(video_file_to_embed)})")
                if not os.path.exists(video_file_to_embed):
                    _append_log(f"ERROR: video file not found!")
                    result['timeline'] = metadata
                    result['error'] = 'Video file not found'
                    return jsonify(result)

                file_size = os.path.getsize(video_file_to_embed)
                _append_log(
                    f"Video file size: {file_size} bytes (~{file_size / 1024 / 1024:.2f} MB)")

                with open(video_file_to_embed, 'rb') as vf:
                    video_data = vf.read()
                    if not video_data:
                        _append_log(
                            f"ERROR: read {len(video_data)} bytes from file")
                        result['timeline'] = metadata
                        result['error'] = 'Video file empty'
                        return jsonify(result)
                    video_b64 = base64.b64encode(video_data).decode('utf-8')

                _append_log(
                    f"Video base64 encoded: {len(video_b64)} chars (~{len(video_b64) * 3 / 4 / 1024 / 1024:.2f} MB base64)")
                result['timeline'] = metadata
                result['video_data'] = video_b64
                result['video_mime'] = 'video/mp4'
                _append_log("Timeline response ready with video_data")
                return jsonify(result)
            except Exception as e:
                _append_log(f"Timeline video encoding error: {e}")
                import traceback
                _append_log(f"Traceback: {traceback.format_exc()}")
                result['timeline'] = metadata
                result['error'] = f'Video encoding error: {str(e)}'
                _append_log(f"Fallback: returning timeline without video")
                return jsonify(result)
        elif visualize and tmp_out:
            if transcode and shutil.which('ffmpeg'):
                try:
                    tmp_mp4 = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.mp4')
                    tmp_mp4.close()
                    try:
                        _register_tmp_file(tmp_mp4.name)
                    except Exception:
                        pass
                    cmd = [
                        'ffmpeg', '-y', '-i', tmp_out.name,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
                        '-movflags', 'faststart', tmp_mp4.name
                    ]
                    proc = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if proc.returncode == 0:
                        _append_log(
                            f"FFMPEG: transcode success {tmp_mp4.name}")
                        return send_file(tmp_mp4.name, as_attachment=True, download_name='detections.mp4', mimetype='video/mp4')
                    else:
                        _append_log(
                            f"FFMPEG ERROR returncode={proc.returncode}: {proc.stderr}")
                except Exception as e:
                    _append_log(f"FFMPEG exception: {e}")
            try:
                _append_response_log(
                    '/detect/video', 200, frames=frame_idx, total=total_detections, model='tf_mobilenet')
            except Exception:
                pass
            try:
                return send_file(tmp_out.name, as_attachment=True, download_name='detections.avi')
            except Exception as e:
                if tmp_mp4 and os.path.exists(tmp_mp4.name):
                    try:
                        _append_response_log('/detect/video', 200, frames=frame_idx,
                                             total=total_detections, model='tf_mobilenet', fallback='mp4')
                    except Exception:
                        pass
                    return send_file(tmp_mp4.name, as_attachment=True, download_name='detections.mp4', mimetype='video/mp4')
                _append_log(f"Failed to send video file (AVI or MP4): {e}")
                return jsonify(result)
        return jsonify(result)
    finally:
        try:
            if tmp_in:
                os.unlink(tmp_in.name)
        except Exception:
            pass


@app.route('/stream/video')
def stream_video():
    global _stream_tracker, _tracker_lock
    camera_url = request.args.get('camera_url')
    if not camera_url:
        return jsonify({'error': 'camera_url query parameter is required for streaming.'}), 400
    frame_step = int(request.args.get('frame_step', 1))
    conf = float(request.args.get('conf', 0.4))
    class_id = int(request.args.get('class_id', -1))

    with _tracker_lock:
        _stream_tracker = CentroidTracker(maxDisappeared=20, maxDistance=80)

    cap = open_video_capture(camera_url)
    if cap is None or not cap.isOpened():
        return jsonify({'error': 'No se pudo abrir la fuente de video para streaming.'}), 500

    def generate():
        frame_idx = 0
        batch_size = 30
        batch_start_time = time.time()
        frames_in_batch = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    import time as time_module
                    time_module.sleep(0.1)
                    continue
                if frame_idx % frame_step == 0:
                    try:
                        frame_start = time.time()
                        # Detectar objetos usando TensorFlow SSD MobileNet
                        dets = detect_objects(
                            frame, conf_threshold=conf, class_id=class_id)

                        with _tracker_lock:
                            dets = _stream_tracker.update(dets)

                        out_frame = draw_boxes(frame, dets)
                        frame_elapsed = time.time() - frame_start
                        frames_in_batch += 1

                        if frames_in_batch >= batch_size:
                            batch_elapsed = time.time() - batch_start_time
                            avg_time_per_frame = batch_elapsed / frames_in_batch
                            _append_log(
                                f"STREAM_TIMING: {frames_in_batch} frames in {batch_elapsed:.2f}s (avg {avg_time_per_frame*1000:.1f}ms/frame)")
                            batch_start_time = time.time()
                            frames_in_batch = 0
                    except Exception:
                        out_frame = frame
                    try:
                        ret2, jpg = cv2.imencode(
                            '.jpg', out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    except Exception:
                        ret2 = False
                        jpg = None
                    if not ret2 or jpg is None:
                        continue
                    chunk = jpg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + f"{len(chunk)}".encode() + b"\r\n\r\n" + chunk + b"\r\n")
                frame_idx += 1
        finally:
            try:
                cap.release()
            except Exception:
                pass
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logs')
def get_logs():
    with _log_lock:
        lines = [l for l in list(_log_buffer) if isinstance(
            l, str) and l.startswith('RESPONSE ')]
    return jsonify({'lines': lines})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
