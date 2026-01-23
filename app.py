from flask import Flask, request, jsonify, send_file, Response, send_from_directory
import numpy as np
import cv2
import base64
import tempfile
import os
from flask_cors import CORS
import threading
import subprocess
import time
import logging
from collections import deque
import shutil
import warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except Exception:
    pass

try:
    from device_config import get_device, print_device_info
    print_device_info()
    device = get_device()
except ImportError:
    print("device_config no disponible, usando CPU")
    device = 'cpu'

if device == 'cuda':
    try:
        from cuda_config import check_cuda
        check_cuda()
    except ImportError:
        print("cuda_config no disponible, usando CPU")
        device = 'cpu'

try:
    import tensorflow as tf
    tf_available = True
    tf_version = tf.__version__
except Exception as e:
    tf_available = False
    tf_version = None
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
log_buffer_size = 1000
_log_buffer = deque(maxlen=log_buffer_size)
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

model_pb = os.path.join('ssd_mobileNet', 'frozen_inference_graph.pb')
labels_path = os.path.join(
    'ssd_mobileNet', 'object_detection_classes_coco.txt')

model_pbtxt = os.path.join(
    'ssd_mobileNet', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

model_image_tensor = 'image_tensor:0'
model_boxes_tensor = 'detection_boxes:0'
model_scores_tensor = 'detection_scores:0'
model_classes_tensor = 'detection_classes:0'
model_num_tensor = 'num_detections:0'


def _load_model_names_from_pbtxt(path):
    """Extrae nombres relevantes del archivo .pbtxt para configurar
    los nombres de tensores usados por la aplicación (si es posible).
    """
    global model_image_tensor, model_boxes_tensor, model_scores_tensor, model_classes_tensor, model_num_tensor
    try:
        if not os.path.exists(path):
            _append_log(f"Model pbtxt not found: {path}")
            return
        text = open(path, 'r', encoding='utf-8', errors='ignore').read()
        import re
        nodes = re.findall(r"node\s*\{([^}]+)\}", text, flags=re.DOTALL)
        for n in nodes:
            mname = re.search(r'name\s*:\s*"([^"]+)"', n)
            mop = re.search(r'op\s*:\s*"([^"]+)"', n)
            if not mname or not mop:
                continue
            name = mname.group(1).strip()
            op = mop.group(1).strip()
            if op.lower() == 'placeholder' and model_image_tensor == 'image_tensor:0':
                model_image_tensor = f"{name}:0"
                _append_log(
                    f"Configured image tensor name from pbtxt: {model_image_tensor}")
            if name == 'detection_boxes':
                model_boxes_tensor = 'detection_boxes:0'
            if name == 'detection_scores':
                model_scores_tensor = 'detection_scores:0'
            if name == 'detection_classes':
                model_classes_tensor = 'detection_classes:0'
            if name == 'num_detections':
                model_num_tensor = 'num_detections:0'
        _append_log(
            f"Model tensor names: image={model_image_tensor} boxes={model_boxes_tensor} scores={model_scores_tensor} classes={model_classes_tensor} num={model_num_tensor}")
    except Exception as e:
        _append_log(f"Error parsing pbtxt for model names: {e}")


_load_model_names_from_pbtxt(model_pbtxt)

model_sess = None
model_graph = None
model_lock = threading.Lock()


def translate_label(label: str) -> str:
    """Traduce etiquetas al español"""
    if label is None:
        return ''
    lbl = str(label).strip().lower()
    mapping = {
        'person': 'persona', 'people': 'persona', 'persona': 'persona',
    }
    return mapping.get(lbl, lbl)


DEFAULT_LABELS = ['sin etiqueta']


def _load_coco_labels(path):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) > 0 and lines[0].lower() != 'sin etiqueta':
                    return ['sin etiqueta'] + lines
                return lines or DEFAULT_LABELS
        else:
            _append_log(f"Labels file not found: {path}")
    except Exception as e:
        _append_log(f"Error loading COCO labels from {path}: {e}")
    return DEFAULT_LABELS


coco_classes = _load_coco_labels(labels_path)


def load_local_model():
    """Carga el frozen graph TensorFlow local y devuelve (session, graph).
    Usa tf.compat.v1 para importar el GraphDef desde `model_pb`.
    """
    global model_sess, model_graph
    if not tf_available:
        _append_log("ERROR: TensorFlow no disponible para cargar modelo local")
        return None, None
    if model_sess is None:
        with model_lock:
            if model_sess is None:
                if not os.path.exists(model_pb):
                    _append_log(f"Local TF model file not found: {model_pb}")
                    return None, None
                try:
                    _append_log(
                        "Cargando modelo SSD MobileNet local (TensorFlow)...")
                    graph_def = tf.compat.v1.GraphDef()
                    with tf.io.gfile.GFile(model_pb, 'rb') as f:
                        graph_def.ParseFromString(f.read())
                    graph = tf.Graph()
                    with graph.as_default():
                        tf.import_graph_def(graph_def, name='')
                    sess = tf.compat.v1.Session(graph=graph)
                    model_graph = graph
                    model_sess = sess
                    _append_log("Modelo local TensorFlow cargado")
                except Exception as e:
                    _append_log(f"Error cargando modelo TF local: {e}")
                    return None, None
    return model_sess, model_graph


def _nms_boxes(boxes, scores, iou_threshold=0.5):
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
    """Detecta objetos usando el modelo local SSD MobileNet (frozen graph).
    - `class_id` : si >=0 filtra por esa clase COCO (1-based index as TF returns), -1 acepta todas.
    - Aplica NMS para eliminar detecciones solapadas.
    """
    if not tf_available:
        _append_log("ERROR: TensorFlow no disponible")
        return []

    try:
        sess, graph = load_local_model()
        if sess is None:
            return []

        h, w = image.shape[:2]
        inp_size = 300

        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            img_rgb = image.copy()
        resized = cv2.resize(img_rgb, (inp_size, inp_size))

        input_np = np.expand_dims(resized, 0)

        with model_lock:
            detection_boxes, detection_scores, detection_classes, _ = sess.run([
                graph.get_tensor_by_name(model_boxes_tensor),
                graph.get_tensor_by_name(model_scores_tensor),
                graph.get_tensor_by_name(model_classes_tensor),
                graph.get_tensor_by_name(model_num_tensor)
            ], feed_dict={graph.get_tensor_by_name(model_image_tensor): input_np})

        results = []
        detection_boxes = detection_boxes[0]
        detection_scores = detection_scores[0]
        detection_classes = detection_classes[0]

        raw_boxes = []
        raw_scores = []
        raw_classes = []
        for i, (box, score, cls) in enumerate(zip(detection_boxes, detection_scores, detection_classes)):
            if score < conf_threshold:
                continue
            cls_int = int(cls)
            if class_id >= 0:
                if cls_int != int(class_id):
                    continue
            y1, x1, y2, x2 = box
            x1_px = float(x1 * w)
            y1_px = float(y1 * h)
            x2_px = float(x2 * w)
            y2_px = float(y2 * h)
            raw_boxes.append([x1_px, y1_px, x2_px, y2_px])
            raw_scores.append(float(score))
            raw_classes.append(cls_int)

        keep_indices = _nms_boxes(raw_boxes, raw_scores, iou_threshold=nms_iou)

        for idx in keep_indices:
            x1_px, y1_px, x2_px, y2_px = raw_boxes[idx]
            x = int(max(0, round(x1_px)))
            y = int(max(0, round(y1_px)))
            bw = int(max(0, round(x2_px - x1_px)))
            bh = int(max(0, round(y2_px - y1_px)))
            cls_int = raw_classes[idx]
            label = coco_classes[cls_int] if 0 <= cls_int < len(
                coco_classes) else str(cls_int)
            results.append({
                'x': x,
                'y': y,
                'w': bw,
                'h': bh,
                'confidence': float(raw_scores[idx]),
                'class_id': int(cls_int),
                'label': translate_label(label)
            })

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
    debug = str(request.args.get('debug', 'false')
                ).lower() in ('1', 'true', 'yes')

    if not tf_available:
        return jsonify({'error': 'TensorFlow not available. Install tensorflow.'}), 500

    raw_info = None
    if debug and not visualize:
        sess, graph = load_local_model()
        if sess is None:
            return jsonify({'error': 'Local TensorFlow model not available for debug.'}), 500
        try:
            inp_size = 300
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img_rgb = img.copy()
            resized = cv2.resize(img_rgb, (inp_size, inp_size))
            input_np = np.expand_dims(resized, 0)
            with model_lock:
                boxes_raw, scores_raw, classes_raw, _ = sess.run([
                    graph.get_tensor_by_name(model_boxes_tensor),
                    graph.get_tensor_by_name(model_scores_tensor),
                    graph.get_tensor_by_name(model_classes_tensor),
                    graph.get_tensor_by_name(model_num_tensor)
                ], feed_dict={graph.get_tensor_by_name(model_image_tensor): input_np})
            boxes_raw = boxes_raw[0].tolist()
            scores_raw = scores_raw[0].tolist()
            classes_raw = classes_raw[0].astype(int).tolist()
            limit = 50
            raw_info = {
                'boxes': boxes_raw[:limit], 'scores': scores_raw[:limit], 'classes': classes_raw[:limit]}
        except Exception as e:
            _append_log(f"DEBUG_RAW_ERROR: {e}")
            raw_info = None

    detections = detect_objects(img, conf_threshold=conf, class_id=class_id)

    elapsed_time = time.time() - start_time

    if visualize:
        out_img = draw_boxes(img, detections)
        _, png = cv2.imencode('.png', out_img)
        elapsed_time = time.time() - start_time
        try:
            model_name = 'local_model'
            _append_response_log('/detect/image', 200, count=len(detections),
                                 model=model_name, time_sec=f"{elapsed_time:.2f}")
        except Exception:
            pass
        return Response(png.tobytes(), mimetype='image/png')
        pass
    response = {'detections': detections, 'count': len(
        detections), 'elapsed_seconds': elapsed_time}
    if raw_info is not None:
        response['raw_detections'] = raw_info
    return jsonify(response)


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
    model_name = 'local_model'
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
            if cap is None:
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
                                     total=total_detections, model=model_name, mode='timeline', visualize=visualize)
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
                    '/detect/video', 200, frames=frame_idx, total=total_detections, model=model_name)
            except Exception:
                pass
            try:
                return send_file(tmp_out.name, as_attachment=True, download_name='detections.avi')
            except Exception as e:
                if tmp_mp4 and os.path.exists(tmp_mp4.name):
                    try:
                        _append_response_log('/detect/video', 200, frames=frame_idx,
                                             total=total_detections, model=model_name, fallback='mp4')
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
    app.run(host='0.0.0.0', port=5501, debug=True)
