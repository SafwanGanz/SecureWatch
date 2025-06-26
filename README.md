# CCTV Surveillance System with Face Recognition

## Overview
The **CCTV Surveillance System with Face Recognition** is a state-of-the-art security solution designed for real-time monitoring, face detection, and recognition. Built with modern computer vision and web technologies, it provides a robust platform for clients seeking enhanced security through intelligent video analysis. The system uses YOLOv5 for object detection, face_recognition for identifying known individuals, and SQLite for persistent data storage, delivering actionable insights via a user-friendly web interface.

This project is ideal for clients in sectors like retail, residential complexes, and corporate environments, offering features such as:
- **Real-Time Video Processing**: Detects and tracks people and objects with color-coded bounding boxes (green for trusted, yellow for warning, red for suspicious).
- **Face Recognition**: Identifies known individuals and stores unknown faces for review.
- **Alert System**: Generates and logs alerts for suspicious activities or unknown faces.
- **Analytics Dashboard**: Provides insights into detection trends and system performance.
- **Person Management**: Allows adding, updating, and deleting known persons with photos and status.

For developers, the project is modular, extensible, and built with industry-standard tools, ensuring ease of maintenance and scalability.

## Project Architecture

### Backend
The backend is a Python-based Flask application with Flask-SocketIO for real-time WebSocket communication. Key components include:

- **Flask Server**: Handles HTTP requests and serves the web interface (`index.html`).
- **YOLOv5**: Performs object detection (e.g., persons, cars) using the `yolov5n.pt` model.
- **Face Recognition**: Utilizes `face_recognition` library with Haar cascades and dlib for face detection and recognition.
- **SQLite Database**: Stores person data (ID, name, status, photo, face encoding, age, mood, etc.) and alerts (type, severity, timestamp, etc.).
- **WebSocket**: Streams video frames and detection data to the frontend in real-time.
- **Motion Detection**: Optimizes processing by analyzing frame differences to trigger detections only when motion is detected.

Key endpoints:
- `/api/cameras`: Lists camera status.
- `/api/person-stats`: Retrieves person detection statistics.
- `/api/known-persons`: Manages known persons (GET, POST, DELETE).
- `/api/alerts`: Fetches recent alerts.
- `/api/analytics`: Provides detection and alert analytics.
- `/api/settings`: Configures system settings (e.g., detection sensitivity, video quality).
- `/api/camera/start` and `/api/camera/stop`: Controls the webcam feed.

### Frontend
The frontend is a single-page HTML application (`index.html`) with JavaScript and Tailwind CSS, providing an intuitive interface for monitoring and management. It communicates with the backend via REST APIs and WebSocket for real-time updates.

## Web Interface Sections

Below are the main sections of the web interface, with descriptions and placeholder image references (replace with actual screenshots):

### 1. Live Video Feed
- **Description**: Displays the real-time webcam feed with bounding boxes around detected objects and faces. Bounding boxes are color-coded: green (trusted), yellow (warning), red (suspicious). Each face includes a label with the person’s name, estimated age, and mood.
- **Image**:
  
  <img src="/screenshots/live-feed.png" alt="Live Video Feed" width="600">  
  
  *Placeholder: Shows a webcam feed with a detected face in a green bounding box labeled "John Doe (Age: 35, Mood: Happy)".*

### 2. Alerts Panel
- **Description**: Lists recent alerts (e.g., "Person detected", "Unknown face detected") with severity, timestamp, and acknowledgment status. Users can acknowledge alerts to mark them as reviewed.
- **Image**:
  
  <img src="/screenshots/alrets-panel.png" alt="Alerts Panel" width="600">
  
  *Placeholder: Displays a table with alerts, including a high-severity alert for an unknown face at 01:17 AM, with an "Acknowledge" button.*

### 3. Person Management
- **Description**: Allows users to view, add, update, or delete known persons. Each entry includes a name, status (trusted, warning, suspicious), photo, and notes. Photos are uploaded as base64-encoded images and stored in the database.
- **Image**:
  
  <img src="/screenshots/person-management.png" alt="person Management" width="600">
  <img src="/screenshots/person-management1.png" alt="person Management" width="600">
  
  *Placeholder: Shows a list of known persons with a form to add a new person, including fields for name, status, photo upload, and notes.*

### 4. Analytics Dashboard
- **Description**: Visualizes detection trends over time (24 hours, 7 days, 30 days) with charts for alert types (high, medium, low) and hourly detection counts. Helps users understand security activity patterns.
- **Image**:
  
   <img src="/screenshots/analytics-dashboard.png" alt="Analytics Dashboard" width="600">
   
  *Placeholder: Displays a bar chart of alerts by severity and a line chart of detections per hour.*

### 5. Settings
- **Description**: Configures system parameters, such as face detection confidence, video quality (low, medium, high), and alert sensitivity. Users can toggle features like face recognition and bounding box display.
- **Image**:
  
  <img src="/screenshots/settings.png" alt="Settings" width="600">
  
  <img src="/screenshots/settings1.png" alt="Settings" width="600">
  
  *Placeholder: Shows a form with sliders for detection confidence, dropdowns for video quality, and checkboxes for enabling/disabling features.*

### 6. System Status
- **Description**: Provides an overview of system health, including uptime, storage usage, active streams, and detected persons. Indicates whether face detection and landmark detection are enabled.
- **Image**:
  
  <img src="/screenshots/system-status.png" alt="System Status" width="600">
  
  *Placeholder: Displays metrics like "Uptime: 99.9%", "Active Streams: 1", and "Total Known Faces: 5".*

## Installation and Setup

### Prerequisites
- **Hardware**:
  - A webcam or IP camera compatible with OpenCV.
  - A computer with at least 8 GB RAM and a modern CPU (GPU recommended for faster YOLOv5 processing).
- **Software**:
  - Python 3.8+ (tested with 3.13).
  - Visual Studio Build Tools (for `dlib` compilation on Windows).
  - A modern web browser (Chrome, Firefox, Edge).
- **Model Files**:
  - `yolov5n.pt`: YOLOv5 model for object detection.
  - `shape_predictor_68_face_landmarks.dat`: dlib model for facial landmark detection.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SafwanGanz/SecureWatch.git
   cd SecureWatch
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - Create a `requirements.txt` file (see below) and run:
     ```bash
     pip install -r requirements.txt
     ```
   - For `dlib`, ensure Visual Studio Build Tools with the C++ workload and `cmake` are installed:
     ```bash
     pip install cmake
     pip install dlib --verbose
     ```

4. **Download Model Files**:
   - **YOLOv5 Model**:
     - Download `yolov5n.pt` from the [YOLOv5 releases page](https://github.com/ultralytics/yolov5/releases).
     - Place it in the project root: `C:\Users\LENOVO\OneDrive\Desktop\votion project\cctv\yolov5n.pt`.
   - **dlib Shape Predictor**:
     - Download `shape_predictor_68_face_landmarks.dat.bz2` from [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
     - Extract the `.bz2` file using 7-Zip or WinRAR to get `shape_predictor_68_face_landmarks.dat` (~99.7 MB).
     - Place it in the project root: `C:\Users\LENOVO\OneDrive\Desktop\votion project\cctv\shape_predictor_68_face_landmarks.dat`.
     - Alternatively, place it in a `models` subdirectory and update `app.py`:
       ```python
       landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
       ```

5. **Create Templates Directory**:
   - Create a `templates` folder in the project root.
   - Place the `index.html` file (provided separately) in `templates/`.

6. **Run the Application**:
   ```bash
   python app.py
   ```
   - Access the web interface at `http://localhost:5000` or `http://<your-ip>:5000` (e.g., `http://192.168.1.106:5000`) from another device on the same network.

### Optional: Production Deployment
For production, use a WSGI server like Waitress:
```bash
pip install waitress
```
Update the `if __name__ == '__main__':` block in `app.py`:
```python
if __name__ == '__main__':
    init_db()
    load_known_faces()
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
```

## Usage
1. **Access the Dashboard**:
   - Open `http://localhost:5000` in a browser.
   - The Live Video Feed starts automatically if the webcam is connected.

2. **Manage Persons**:
   - Navigate to the Person Management section to add new persons with names, statuses, and photos.
   - Update or delete existing persons as needed.

3. **Monitor Alerts**:
   - Check the Alerts Panel for real-time notifications of detected objects or faces.
   - Acknowledge alerts to mark them as reviewed.

4. **Analyze Trends**:
   - Use the Analytics Dashboard to view detection statistics over different time periods.

5. **Adjust Settings**:
   - Modify detection confidence, video quality, or enable/disable features in the Settings section.

## Developer Notes
- **Extensibility**:
  - Add new detection models by updating the YOLOv5 model path or integrating other frameworks like TensorFlow.
  - Extend the database schema to store additional metadata (e.g., location, time zones).
- **Optimization**:
  - Optimize video processing by adjusting `detection_fps` or `stream_fps` in `settings`.
  - Implement GPU acceleration for YOLOv5 if available.
- **Error Handling**:
  - The application includes robust error handling for camera failures and missing model files.
  - Logs are written with timestamps for debugging (see `logging` configuration).

## Troubleshooting
- **Camera Not Found**:
  - Ensure the webcam is connected and not used by another application.
  - Check logs for `Failed to initialize webcam` errors.
- **Model File Missing**:
  - Verify `yolov5n.pt` and `shape_predictor_68_face_landmarks.dat` are in the project root or update paths in `app.py`.
- **URI Too Long (414 Error)**:
  - Ensure frontend GET requests (e.g., `/api/known-persons`) do not include large data in the URL.
  - Compress images in the `/api/known-persons` POST endpoint (already implemented in `app.py`).
- **Serialization Errors**:
  - The `convert_numpy_types` function handles NumPy and bool types to prevent JSON serialization issues.

## Links and Resources
- **YOLOv5**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **dlib Shape Predictor**: [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- **Flask**: [https://flask.palletsprojects.com](https://flask.palletsprojects.com)
- **Flask-SocketIO**: [https://flask-socketio.readthedocs.io](https://flask-socketio.readthedocs.io)
- **face_recognition**: [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- **OpenCV**: [https://opencv.org](https://opencv.org)

## Contact
For support or customization, contact the development team at [safwanganz@gmail.com](mailto:safwanganz@gmail.com).
