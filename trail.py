from flask import Flask, render_template, request, jsonify, Response
import cv2
import torch
import threading
import queue
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.7
    model.iou = 0.5
    model.max_det = 20
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Known object widths in cm (updated with more accurate measurements)
KNOWN_WIDTHS = {
    'person': 45,  # Average shoulder width
    'bottle': 8,
    'cup': 7,
    'chair': 45,  # Average chair width
    'table': 80,  # Average table width
    'car': 180,   # Average car width
    'bicycle': 60, # Average bicycle width
    'motorcycle': 80, # Average motorcycle width
    'backpack': 30, # Average backpack width
    'handbag': 25,  # Average handbag width
    'umbrella': 80, # Average umbrella width
    'dog': 40,     # Average dog width
    'cat': 25,     # Average cat width
    'tv': 100,     # Average TV width
    'laptop': 35,  # Average laptop width
    'keyboard': 45, # Average keyboard width
    'mouse': 6,    # Average mouse width
    'cell phone': 7, # Average phone width
    'book': 15,    # Average book width
    'clock': 30,   # Average clock width
    'wall': 200,    # Average wall width for detection
}

FOCAL_LENGTH = 650
MAX_DISTANCE = 75
MIN_DISTANCE = 30
DISTANCE_CORRECTION_FACTOR = 1.15
MIN_CONFIDENCE = 0.7
MIN_OBJECT_SIZE = 50

detection_queue = queue.Queue()
is_detection_running = False
detection_thread = None
last_detection_time = 0
DETECTION_COOLDOWN = 0.15
last_detections = []

def calculate_distance(known_width, pixel_width, confidence, height_pixels):
    if pixel_width == 0 or height_pixels == 0:
        return float('inf')
    try:
        aspect_ratio = height_pixels / pixel_width
        base_distance = (known_width * FOCAL_LENGTH) / pixel_width
        confidence_factor = 0.7 + (confidence * 0.6)
        aspect_ratio_factor = 1.0
        if 0.6 <= aspect_ratio <= 1.8:
            aspect_ratio_factor = 1.0 + (1.0 - min(abs(1.0 - aspect_ratio), 1.0)) * 0.3
        corrected_distance = base_distance * confidence_factor * aspect_ratio_factor * DISTANCE_CORRECTION_FACTOR
        if corrected_distance < MIN_DISTANCE or corrected_distance > MAX_DISTANCE:
            return float('inf')
        return corrected_distance
    except Exception as e:
        logger.error(f"Error calculating distance: {str(e)}")
        return float('inf')

def smooth_detections(new_detections):
    global last_detections
    if not last_detections:
        last_detections = new_detections
        return new_detections
    smoothed_detections = []
    for new_det in new_detections:
        matching_det = next((d for d in last_detections 
                           if d['class'] == new_det['class'] 
                           and d['position'] == new_det['position']), None)
        if matching_det:
            smoothed_distance = int(0.7 * new_det['distance'] + 0.3 * matching_det['distance'])
            new_det['distance'] = smoothed_distance
        smoothed_detections.append(new_det)
    last_detections = smoothed_detections
    return smoothed_detections

def process_frame(frame_data):
    global last_detection_time
    current_time = time.time()
    if current_time - last_detection_time < DETECTION_COOLDOWN:
        return []
    last_detection_time = current_time
    try:
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Failed to decode frame")
            return []
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        results = model(frame)
        detections = results.xyxy[0]
        current_detections = []
        detected_objects = set()
        for *box, conf, cls in detections:
            class_name = model.names[int(cls)]
            if class_name in KNOWN_WIDTHS and conf >= MIN_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box)
                width_pixels = x2 - x1
                height_pixels = y2 - y1
                if width_pixels < MIN_OBJECT_SIZE or height_pixels < MIN_OBJECT_SIZE:
                    continue
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                distance = calculate_distance(KNOWN_WIDTHS[class_name], width_pixels, conf, height_pixels)
                if MIN_DISTANCE <= distance <= MAX_DISTANCE:
                    left_third = frame_width / 3
                    right_third = 2 * frame_width / 3
                    if center_x < left_third:
                        position = "left"
                    elif center_x > right_third:
                        position = "right"
                    else:
                        position = "front"
                    object_id = f"{class_name}_{position}_{int(distance)}"
                    if object_id not in detected_objects:
                        detected_objects.add(object_id)
                        current_detections.append({
                            'class': class_name,
                            'distance': int(distance),
                            'position': position,
                            'confidence': float(conf),
                            'size': {
                                'width': width_pixels,
                                'height': height_pixels
                            }
                        })
        current_detections.sort(key=lambda x: x['distance'])
        smoothed_detections = smooth_detections(current_detections)
        return smoothed_detections
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('ipFront2.html')

@app.route('/process-frame', methods=['POST'])
def process_frame_route():
    if not is_detection_running:
        return jsonify({'error': 'Detection not running'})
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'})
    try:
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        detections = process_frame(frame_data)
        return jsonify({'detections': detections})
    except Exception as e:
        logger.error(f"Error in process_frame_route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/run-detection')
def start_detection():
    global is_detection_running, last_detections
    if not is_detection_running:
        is_detection_running = True
        last_detections = []
        logger.info("Detection started")
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already running'})

@app.route('/stop-detection')
def stop_detection():
    global is_detection_running, last_detections
    is_detection_running = False
    last_detections = []
    logger.info("Detection stopped")
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
