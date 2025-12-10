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
    from ultralytics import YOLO
except Exception:
    YOLO = None


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
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
yolo_model = None
yolo_lock = threading.Lock()
ssd_net = None
ssd_lock = threading.Lock()
ssd_class_names = []
SSD_DIR = os.path.join(os.path.dirname(__file__), 'ssd_mobileNet')
SSD_PB = os.path.join(SSD_DIR, 'frozen_inference_graph.pb')
SSD_PBTXT = os.path.join(SSD_DIR, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
SSD_CLASSES = os.path.join(SSD_DIR, 'object_detection_classes_coco.txt')


def _load_ssd_class_names():
    global ssd_class_names
    try:
        if os.path.exists(SSD_CLASSES):
            with open(SSD_CLASSES, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                ssd_class_names = lines
    except Exception:
        ssd_class_names = []


def get_ssd_model():
    global ssd_net
    if ssd_net is None:
        with ssd_lock:
            if ssd_net is None:
                try:
                    if os.path.exists(SSD_PB) and os.path.exists(SSD_PBTXT):
                        net = cv2.dnn.readNetFromTensorflow(SSD_PB, SSD_PBTXT)
                        ssd_net = net
                    else:
                        _append_log(f"SSD model files not found in {SSD_DIR}")
                except Exception as e:
                    _append_log(f"Error loading SSD model: {e}")
    if not ssd_class_names:
        _load_ssd_class_names()
    return ssd_net


def detect_people_ssd(image, conf_threshold=0.5):
    model = get_ssd_model()
    if model is None:
        return []
    try:
        net = model
        if net is None:
            return []
        h, w = image.shape[:2]
        inp_size = 300
        blob = cv2.dnn.blobFromImage(image, 1.0, size=(
            inp_size, inp_size), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        detections_raw = net.forward()
        detections = []
        total = int(detections_raw.shape[2])
        for i in range(total):
            cls = int(detections_raw[0, 0, i, 1])
            score = float(detections_raw[0, 0, i, 2])
            if score < float(conf_threshold):
                continue

            if cls != 1:
                continue

            x1 = float(detections_raw[0, 0, i, 3]) * w
            y1 = float(detections_raw[0, 0, i, 4]) * h
            x2 = float(detections_raw[0, 0, i, 5]) * w
            y2 = float(detections_raw[0, 0, i, 6]) * h
            x = int(max(0, x1))
            y = int(max(0, y1))
            bw = int(max(0, x2 - x1))
            bh = int(max(0, y2 - y1))
            name = ''
            try:
                if ssd_class_names and cls < len(ssd_class_names):
                    name = ssd_class_names[cls]
                else:
                    name = str(cls)
            except Exception:
                name = str(cls)
            label = translate_label(name)
            detections.append({'x': x, 'y': y, 'w': bw, 'h': bh, 'confidence': float(
                score), 'class_id': cls, 'label': label})
        return detections
    except Exception as e:
        _append_log(f"SSD detection error: {e}")
        return []


def translate_label(label: str) -> str:
    if label is None:
        return ''
    lbl = str(label).strip().lower()
    mapping = {
        'person': 'persona', 'people': 'persona', 'persona': 'persona', 'personas': 'persona',
        'car': 'coche', 'bicycle': 'bicicleta', 'motorcycle': 'moto', 'truck': 'camión', 'bus': 'autobús',
        'dog': 'perro', 'cat': 'gato', 'chair': 'silla', 'bench': 'banco', 'bird': 'pájaro',
        'boat': 'barco', 'traffic light': 'semáforo', 'backpack': 'mochila', 'umbrella': 'paraguas'
    }
    return mapping.get(lbl, lbl)


def get_yolo_model():
    global yolo_model
    if YOLO is None:
        return None
    if yolo_model is None:
        with yolo_lock:
            if yolo_model is None:
                yolo_model = YOLO('yolo11n.pt')
                try:
                    yolo_model.to(DEVICE)
                    logging.info(f"Modelo YOLOv11 cargado en {DEVICE.upper()}")
                except:
                    logging.warning(
                        f"No se pudo mover el modelo a {DEVICE}, usando CPU")
                    yolo_model.to('cpu')
    return yolo_model


def detect_people(image, win_stride=(8, 8), padding=(8, 8), scale=1.05):
    rects, weights = hog.detectMultiScale(
        image, winStride=win_stride, padding=padding, scale=scale)
    detections = []
    for (x, y, w, h), weight in zip(rects, weights):
        conf = float(weight) if hasattr(weight, 'item') else float(weight)
        detections.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "confidence": conf,
            "label": translate_label("person")
        })
    return detections


def detect_people_yolo(image, conf_threshold=0.25):
    model = get_yolo_model()
    if model is None:
        return []
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, conf=conf_threshold, verbose=False)
    detections = []
    if len(results) == 0:
        return detections
    res = results[0]
    boxes = getattr(res, 'boxes', None)
    if boxes is None:
        return detections
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
    except Exception:
        xyxy = np.array(boxes.xyxy)
        confs = np.array(boxes.conf)
        clss = np.array(boxes.cls)
    names = None
    try:
        names = model.names if hasattr(model, 'names') else None
    except Exception:
        names = None
    for idx in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[idx]
        conf = confs[idx]
        cls = clss[idx]
        cls_i = int(cls)

        if cls_i != 0:
            continue

        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        label = None
        if names and cls_i in names:
            try:
                label = names[cls_i]
            except Exception:
                label = str(names.get(cls_i, cls_i))
        else:
            label = str(cls_i)

        label = translate_label(label)

        detections.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'confidence': float(conf),
            'class_id': cls_i,
            'label': label
        })
    return detections


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
    model_name = request.args.get('model', 'hog').lower()
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
    conf = float(request.args.get('conf', 0.25))
    if model_name == 'yolo':
        detections = detect_people_yolo(img, conf_threshold=conf)
        if detections == [] and YOLO is None:
            return jsonify({'error': 'YOLO model not available. Install ultralytics or check logs.'}), 500
    elif model_name == 'ssd' or model_name == 'mobilenet':
        detections = detect_people_ssd(img, conf_threshold=conf)
    else:
        detections = detect_people(img)
    if visualize:
        out_img = draw_boxes(img, detections)
        _, png = cv2.imencode('.png', out_img)
        elapsed_time = time.time() - start_time
        try:
            _append_response_log('/detect/image', 200,
                                 count=len(detections), model=model_name, time_sec=f"{elapsed_time:.2f}")
        except Exception:
            pass
        return Response(png.tobytes(), mimetype='image/png')
    elapsed_time = time.time() - start_time
    try:
        _append_response_log('/detect/image', 200,
                             count=len(detections), model=model_name, time_sec=f"{elapsed_time:.2f}")
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
    model_name = request.form.get(
        'model', request.args.get('model', 'hog')).lower()
    conf = float(request.form.get('conf', request.args.get('conf', 0.25)))
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
                if model_name == 'yolo':
                    dets = detect_people_yolo(frame, conf_threshold=conf)
                elif model_name == 'ssd' or model_name == 'mobilenet':
                    dets = detect_people_ssd(frame, conf_threshold=conf)
                else:
                    dets = detect_people(frame)

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
    model_name = request.args.get('model', 'hog').lower()
    frame_step = int(request.args.get('frame_step', 1))
    conf = float(request.args.get('conf', 0.25))

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
                        if model_name == 'yolo':
                            dets = detect_people_yolo(
                                frame, conf_threshold=conf)
                        elif model_name == 'ssd' or model_name == 'mobilenet':
                            dets = detect_people_ssd(
                                frame, conf_threshold=conf)
                        else:
                            dets = detect_people(frame)

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
