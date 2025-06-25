from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime
import random
import uuid
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
import threading
import time
import logging
from collections import deque, defaultdict
import dlib
import face_recognition

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO model once and set to eval mode for better performance
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', force_reload=True)
yolo_model.eval()

# Face detection and landmark detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Error loading Haar cascade classifier")

# Initialize dlib face detector and landmark predictor
try:
    # Download shape_predictor_68_face_landmarks.dat from dlib's website if not present
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    landmark_detection_available = True
    logging.info("Facial landmark detection initialized successfully")
except Exception as e:
    logging.warning(f"Facial landmark detection not available: {e}")
    landmark_detection_available = False

cap = None
camera_thread = None
camera_running = False
camera_initialized = False

# Performance optimization variables
frame_count = 0
last_detection_time = 0
detection_interval = 0.5
last_frame_time = 0
target_fps = 15
min_frame_interval = 1.0 / target_fps

# Motion detection for smart processing
last_frame_gray = None
motion_threshold = 25

# Face detection and recognition variables
last_face_detection_time = 0
face_detection_interval = 1.0
known_faces = []  # Store known face encodings
known_face_names = []  # Store corresponding names
person_stats = defaultdict(lambda: {
    'first_seen': None,
    'last_seen': None,
    'total_detections': 0,
    'avg_confidence': 0.0,
    'face_landmarks': [],
    'emotions': [],
    'age_estimates': [],
    'is_known': False,
    'name': 'Unknown'
})

settings = {
    'privacy_mode': False,
    'alert_sensitivity': 'medium',
    'motion_threshold': 50,
    'notification_enabled': True,
    'yolo_confidence': 0.5,
    'video_quality': 'medium',
    'detection_fps': 2,
    'stream_fps': 15,
    'face_detection_enabled': True,
    'face_detection_confidence': 1.3,
    'face_min_neighbors': 5,
    'face_min_size': (30, 30),
    'show_bounding_boxes': True,
    'show_facial_landmarks': True,
    'face_recognition_enabled': True,
    'person_tracking_enabled': True,
    'save_unknown_faces': False
}

alerts = []

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def initialize_webcam():
    global cap, camera_initialized
    if cap is not None:
        cap.release()
        cap = None
    
    for index in range(3):
        try:
            test_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    logging.info(f"Webcam opened successfully on index {index}")
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                    test_cap.set(cv2.CAP_PROP_FPS, target_fps)
                    test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap = test_cap
                    camera_initialized = True
                    return True
                else:
                    logging.warning(f"Camera at index {index} opened but cannot read frames")
            test_cap.release()
        except Exception as e:
            logging.error(f"Error testing camera at index {index}: {e}")
    
    logging.error("Failed to open webcam on indices 0, 1, or 2")
    camera_initialized = False
    cap = None
    return False

def detect_motion(current_frame, last_frame, threshold=25):
    """Detect motion between frames to skip processing when nothing changes"""
    if last_frame is None:
        return True, current_frame
    
    diff = cv2.absdiff(current_frame, last_frame)
    mean_diff = np.mean(diff)
    
    return mean_diff > threshold, current_frame

def resize_frame_for_detection(frame, max_size=416):
    """Resize frame for faster YOLO detection while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height)), scale
    return frame, 1.0

def get_facial_landmarks(gray_frame, face_rect):
    """Extract 68 facial landmarks from a face"""
    if not landmark_detection_available:
        return []
    
    try:
        # Convert OpenCV rectangle to dlib rectangle
        dlib_rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[2], face_rect[3])
        landmarks = landmark_predictor(gray_frame, dlib_rect)
        
        # Convert landmarks to list of (x, y) coordinates
        landmark_points = []
        for i in range(68):
            point = landmarks.part(i)
            landmark_points.append([point.x, point.y])
        
        return landmark_points
    except Exception as e:
        logging.error(f"Error extracting facial landmarks: {e}")
        return []

def recognize_face(face_encoding):
    """Recognize a face using face_recognition library"""
    if not settings['face_recognition_enabled'] or len(known_faces) == 0:
        return "Unknown", 0.0
    
    try:
        # Compare face encoding with known faces
        distances = face_recognition.face_distance(known_faces, face_encoding)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        
        # If distance is below threshold, it's a match
        if min_distance < 0.6:  # Adjust threshold as needed
            confidence = 1.0 - min_distance
            return known_face_names[min_distance_index], confidence
        else:
            return "Unknown", 0.0
    except Exception as e:
        logging.error(f"Error in face recognition: {e}")
        return "Unknown", 0.0

def detect_faces_enhanced(frame, gray_frame):
    """Enhanced face detection with landmarks and recognition"""
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=settings['face_detection_confidence'],
        minNeighbors=settings['face_min_neighbors'],
        minSize=settings['face_min_size'],
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_data = []
    current_time = datetime.now()
    
    for (x, y, w, h) in faces:
        try:
            # Extract face region for encoding
            face_region = frame[y:y+h, x:x+w]
            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get face encoding for recognition
            face_encoding = None
            person_name = "Unknown"
            recognition_confidence = 0.0
            
            if settings['face_recognition_enabled']:
                try:
                    face_encodings = face_recognition.face_encodings(face_region_rgb)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        person_name, recognition_confidence = recognize_face(face_encoding)
                except Exception as e:
                    logging.debug(f"Face encoding failed: {e}")
            
            # Calculate detection confidence based on face size
            area = w * h
            detection_confidence = min(0.9, 0.5 + (area / 10000))
            
            # Get facial landmarks
            landmarks = []
            if settings['show_facial_landmarks']:
                landmarks = get_facial_landmarks(gray_frame, [x, y, x+w, y+h])
            
            # Create unique person ID based on face position and encoding
            person_id = f"person_{hash(str(face_encoding))}" if face_encoding is not None else f"person_{x}_{y}"
            
            # Update person statistics
            if settings['person_tracking_enabled']:
                stats = person_stats[person_id]
                if stats['first_seen'] is None:
                    stats['first_seen'] = current_time.isoformat()
                stats['last_seen'] = current_time.isoformat()
                stats['total_detections'] += 1
                stats['avg_confidence'] = (stats['avg_confidence'] * (stats['total_detections'] - 1) + detection_confidence) / stats['total_detections']
                stats['face_landmarks'] = landmarks
                stats['is_known'] = recognition_confidence > 0.6
                stats['name'] = person_name
            
            face_info = {
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'label': 'face',
                'confidence': float(detection_confidence),
                'width': int(w),
                'height': int(h),
                'type': 'face',
                'person_id': person_id,
                'person_name': person_name,
                'recognition_confidence': float(recognition_confidence),
                'landmarks': landmarks,
                'is_known': recognition_confidence > 0.6
            }
            
            face_data.append(face_info)
            
        except Exception as e:
            logging.error(f"Error processing face: {e}")
    
    return face_data

def draw_bounding_boxes_and_landmarks(frame, detections):
    """Draw bounding boxes and facial landmarks on frame for preview"""
    if not settings['show_bounding_boxes']:
        return frame
        
    annotated_frame = frame.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on detection type
        if detection['type'] == 'face':
            color = (0, 255, 0) if detection.get('is_known', False) else (0, 255, 255)  # Green for known, yellow for unknown
            label = f"{detection.get('person_name', 'Unknown')} ({detection['confidence']:.2f})"
        else:
            color = (255, 0, 0)  # Blue for objects
            label = f"{detection['label']} ({detection['confidence']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw facial landmarks if available
        if settings['show_facial_landmarks'] and detection['type'] == 'face' and detection.get('landmarks'):
            landmarks = detection['landmarks']
            for point in landmarks:
                cv2.circle(annotated_frame, tuple(map(int, point)), 1, (0, 0, 255), -1)
    
    return annotated_frame

def process_video():
    global cap, camera_running, camera_initialized, frame_count, last_detection_time
    global last_frame_time, last_frame_gray, last_face_detection_time
    
    if not camera_initialized or cap is None:
        if not initialize_webcam():
            socketio.emit('error', {'message': 'Failed to initialize webcam. Please check camera connection.'})
            camera_running = False
            return

    cached_detections = []
    cached_faces = []
    detection_cache_time = 0
    face_cache_time = 0

    while camera_running:
        try:
            current_time = time.time()
            
            if current_time - last_frame_time < min_frame_interval:
                time.sleep(0.01)
                continue
            
            if cap is None:
                logging.error("Camera capture object is None")
                socketio.emit('error', {'message': 'Camera capture failed. Attempting to reinitialize...'})
                if not initialize_webcam():
                    camera_running = False
                    break
                continue
            
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from webcam")
                socketio.emit('error', {'message': 'Failed to read webcam frame. Camera may be disconnected.'})
                if not initialize_webcam():
                    camera_running = False
                    break
                continue

            last_frame_time = current_time
            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            has_motion, last_frame_gray = detect_motion(gray, last_frame_gray, motion_threshold)
            
            detection_data = []
            face_data = []
            
            # YOLO Detection
            if has_motion and (current_time - last_detection_time) > (1.0 / settings['detection_fps']):
                last_detection_time = current_time
                
                detection_frame, scale = resize_frame_for_detection(frame)
                frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    with torch.no_grad():
                        results = yolo_model(frame_rgb)
                
                detections = results.xyxy[0].cpu().numpy()
                
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if conf >= settings['yolo_confidence']:
                        label = yolo_model.names[int(cls)]
                        detection_data.append({
                            'bbox': [float(x1/scale), float(y1/scale), float(x2/scale), float(y2/scale)],
                            'label': label,
                            'confidence': float(conf),
                            'type': 'yolo'
                        })

                        if label in ['person', 'car'] and random.random() < 0.05:
                            alert = {
                                'id': str(uuid.uuid4()),
                                'camera_id': 'webcam_001',
                                'camera_name': 'Webcam',
                                'type': f'{label}_detected',
                                'severity': 'medium' if float(conf) < 0.7 else 'high',
                                'message': f'{label.capitalize()} detected in frame',
                                'timestamp': datetime.now().isoformat(),
                                'acknowledged': False
                            }
                            alerts.insert(0, alert)
                            socketio.emit('new_alert', alert)
                
                cached_detections = detection_data
                detection_cache_time = current_time
            else:
                if current_time - detection_cache_time < 1.0:
                    detection_data = cached_detections

            # Enhanced Face Detection
            if (settings['face_detection_enabled'] and 
                has_motion and 
                (current_time - last_face_detection_time) > face_detection_interval):
                
                last_face_detection_time = current_time
                face_data = detect_faces_enhanced(frame, gray)
                
                if face_data and random.random() < 0.1:
                    known_faces_count = sum(1 for f in face_data if f.get('is_known', False))
                    unknown_faces_count = len(face_data) - known_faces_count
                    
                    alert = {
                        'id': str(uuid.uuid4()),
                        'camera_id': 'webcam_001',
                        'camera_name': 'Webcam',
                        'type': 'face_detected',
                        'severity': 'high' if unknown_faces_count > 0 else 'medium',
                        'message': f'{len(face_data)} face(s) detected: {known_faces_count} known, {unknown_faces_count} unknown',
                        'timestamp': datetime.now().isoformat(),
                        'acknowledged': False,
                        'face_count': len(face_data),
                        'known_faces': known_faces_count,
                        'unknown_faces': unknown_faces_count
                    }
                    alerts.insert(0, alert)
                    socketio.emit('new_alert', alert)
                
                cached_faces = face_data
                face_cache_time = current_time
            else:
                if current_time - face_cache_time < 2.0:
                    face_data = cached_faces

            all_detections = detection_data + face_data
            
            # Draw annotations on frame
            annotated_frame = draw_bounding_boxes_and_landmarks(frame, all_detections)
            
            quality_map = {'low': 50, 'medium': 70, 'high': 85}
            jpeg_quality = quality_map.get(settings['video_quality'], 70)
            
            height, width = annotated_frame.shape[:2]
            
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            frame_data = {
                'image': jpg_as_text,
                'detections': convert_numpy_types(all_detections),
                'yolo_detections': convert_numpy_types(detection_data),
                'face_detections': convert_numpy_types(face_data),
                'width': int(width),
                'height': int(height),
                'frame_count': frame_count,
                'motion_detected': bool(has_motion),
                'face_count': len(face_data),
                'known_faces': len([f for f in face_data if f.get('is_known', False)]),
                'unknown_faces': len([f for f in face_data if not f.get('is_known', False)])
            }
            
            socketio.emit('video_frame', frame_data)
            
            if frame_count % 30 == 0:
                # Send person statistics
                stats_summary = {}
                for person_id, stats in person_stats.items():
                    if stats['last_seen'] and (datetime.now() - datetime.fromisoformat(stats['last_seen'])).seconds < 300:  # Last 5 minutes
                        stats_summary[person_id] = {
                            'name': stats['name'],
                            'total_detections': stats['total_detections'],
                            'avg_confidence': stats['avg_confidence'],
                            'is_known': stats['is_known'],
                            'first_seen': stats['first_seen'],
                            'last_seen': stats['last_seen']
                        }
                
                socketio.emit('heartbeat', {
                    'status': 'alive', 
                    'fps': settings['stream_fps'],
                    'face_detection_enabled': settings['face_detection_enabled'],
                    'person_stats': stats_summary
                })
            
        except Exception as e:
            logging.error(f"Error in video processing: {e}")
            socketio.emit('error', {'message': f'Video processing error: {str(e)}'})
            time.sleep(0.1)

    if cap is not None:
        cap.release()
        cap = None
    camera_initialized = False
    logging.info("Video processing stopped")

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/cameras')
def get_cameras():
    global camera_initialized
    camera = {
        'id': 'webcam_001',
        'name': 'Webcam',
        'location': 'Local Device',
        'status': 'online' if camera_running and camera_initialized else 'offline',
        'last_seen': datetime.now().isoformat(),
        'face_detection_enabled': settings['face_detection_enabled'],
        'landmark_detection_available': landmark_detection_available
    }
    return jsonify({
        'cameras': [camera],
        'total': 1,
        'online': 1 if camera_running and camera_initialized else 0
    })

@app.route('/api/person-stats')
def get_person_stats():
    """Get statistics for all detected persons"""
    return jsonify({
        'person_stats': convert_numpy_types(dict(person_stats)),
        'total_persons': len(person_stats),
        'known_persons': len([p for p in person_stats.values() if p['is_known']]),
        'unknown_persons': len([p for p in person_stats.values() if not p['is_known']])
    })

@app.route('/api/alerts')
def get_alerts():
    return jsonify({
        'alerts': alerts[:10],
        'total': len(alerts),
        'unacknowledged': len([a for a in alerts if not a['acknowledged']])
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global settings
    if request.method == 'GET':
        return jsonify(settings)
    
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update all existing settings
        for key, value in data.items():
            if key in settings:
                if key in ['privacy_mode', 'notification_enabled', 'face_detection_enabled', 
                          'show_bounding_boxes', 'show_facial_landmarks', 'face_recognition_enabled',
                          'person_tracking_enabled', 'save_unknown_faces']:
                    settings[key] = bool(value)
                elif key in ['motion_threshold', 'face_min_neighbors']:
                    settings[key] = int(value)
                elif key in ['yolo_confidence', 'face_detection_confidence', 'detection_fps']:
                    settings[key] = float(value)
                elif key in ['stream_fps']:
                    settings[key] = int(value)
                    global target_fps, min_frame_interval
                    target_fps = int(value)
                    min_frame_interval = 1.0 / target_fps
                else:
                    settings[key] = value

        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'settings': settings
        })

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    for alert in alerts:
        if alert['id'] == alert_id:
            alert['acknowledged'] = True
            socketio.emit('alert_updated', alert)
            return jsonify({'success': True, 'message': 'Alert acknowledged'})
    return jsonify({'error': 'Alert not found'}), 404

@app.route('/api/system/status')
def system_status():
    return jsonify({
        'system_status': 'operational',
        'uptime': '99.9%',
        'storage_used': random.randint(45, 85),
        'bandwidth_usage': random.randint(20, 60),
        'active_streams': 1 if camera_running and camera_initialized else 0,
        'current_fps': settings['stream_fps'],
        'detection_fps': settings['detection_fps'],
        'face_detection_enabled': settings['face_detection_enabled'],
        'landmark_detection_available': landmark_detection_available,
        'total_known_faces': len(known_faces),
        'active_persons': len([p for p in person_stats.values() if p['last_seen'] and 
                              (datetime.now() - datetime.fromisoformat(p['last_seen'])).seconds < 300])
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=process_video)
        camera_thread.daemon = True
        camera_thread.start()
        logging.info("Camera start requested")
        return jsonify({'success': True, 'message': 'Camera started'})
    logging.warning("Camera already running")
    return jsonify({'success': False, 'message': 'Camera already running'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera_running, cap
    if camera_running:
        camera_running = False
        time.sleep(0.2)
        if cap is not None:
            cap.release()
            cap = None
        logging.info("Camera stopped")
        return jsonify({'success': True, 'message': 'Camera stopped'})
    logging.warning("Camera not running")
    return jsonify({'success': False, 'message': 'Camera not running'})

@socketio.on('connect')
def handle_connect():
    logging.info("WebSocket client connected")
    emit('connection_status', {'status': 'connected'})

@socketio.on('heartbeat')
def handle_heartbeat(data):
    logging.debug("Heartbeat received")

@socketio.on('error')
def handle_error(data):
    logging.error(f"WebSocket error: {data}")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)