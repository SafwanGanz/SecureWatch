class WebSyncApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.cameras = [];
        this.alerts = [];
        this.settings = {};
        this.updateInterval = null;
        this.socket = io();
        this.isRecording = false;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.cameraStatus = 'offline';
        this.init();
    }
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupNavigation();
        this.loadInitialData();
        this.startAutoRefresh();
    }
    setupWebSocket() {
        this.socket.on('connect', () => {
            this.showNotification('Connected to server', 'success');
            console.log('WebSocket connected');
        });
        this.socket.on('video_frame', (data) => {
            console.log('Received video frame', data);
            const video = document.getElementById('webcam');
            video.src = `data:image/jpeg;base64,${data.image}`;
            video.width = data.width || 1280;
            video.height = data.height || 720;
            video.play().catch(e => console.error('Autoplay failed:', e));
            this.renderDetections(data);
            this.cameraStatus = 'online';
            this.updateCameraStatusUI();
        });
        this.socket.on('new_alert', (alert) => {
            this.alerts.unshift(alert);
            this.renderRecentAlerts();
            this.renderAlertsList();
            this.updateAlertStats({ unacknowledged: this.alerts.filter(a => !a.acknowledged).length });
        });
        this.socket.on('alert_updated', (alert) => {
            const index = this.alerts.findIndex(a => a.id === alert.id);
            if (index !== -1) {
                this.alerts[index] = alert;
                this.renderRecentAlerts();
                this.renderAlertsList();
                this.updateAlertStats({ unacknowledged: this.alerts.filter(a => !a.acknowledged).length });
            }
        });
        this.socket.on('error', (data) => {
            this.showNotification(data.message, 'error');
            this.cameraStatus = 'offline';
            this.updateCameraStatusUI();
            console.error('WebSocket error:', data.message);
        });
        this.socket.on('heartbeat', (data) => {
            console.log('Heartbeat received:', data.status);
        });
    }
    setupEventListeners() {
        document.getElementById('menuToggle')?.addEventListener('click', () => {
            document.getElementById('sidebar')?.classList.remove('-translate-x-full');
            document.getElementById('overlay')?.classList.remove('hidden');
        });
        document.getElementById('overlay')?.addEventListener('click', () => {
            document.getElementById('sidebar')?.classList.add('-translate-x-full');
            document.getElementById('overlay')?.classList.add('hidden');
        });
        document.getElementById('startCamera')?.addEventListener('click', () => this.startWebcam());
        document.getElementById('stopCamera')?.addEventListener('click', () => this.stopWebcam());
        document.getElementById('captureSnapshot')?.addEventListener('click', () => this.captureSnapshot());
        document.getElementById('toggleRecording')?.addEventListener('click', () => this.toggleRecording());
        document.getElementById('saveSettings')?.addEventListener('click', () => this.saveSettings());
        document.getElementById('motionThreshold')?.addEventListener('input', (e) => {
            document.getElementById('motionThresholdValue')?.textContent = e.target.value;
        });
        document.getElementById('yoloConfidence')?.addEventListener('input', (e) => {
            document.getElementById('yoloConfidenceValue')?.textContent = e.target.value;
        });
        ['privacyMode', 'faceBlurring', 'notificationEnabled', 'alertSensitivity', 'videoQuality'].forEach(id => {
            document.getElementById(id)?.addEventListener('change', () => this.updateSettingsUI());
        });
        ['filterAll', 'filterUnacknowledged', 'filterHighPriority'].forEach(id => {
            document.getElementById(id)?.addEventListener('click', () => this.filterAlerts(id));
        });
        document.getElementById('clearAlerts')?.addEventListener('click', () => this.clearAlerts());
    }
    async startWebcam() {
        const video = document.getElementById('webcam');
        if (!video) {
            console.error('Video element not found');
            return;
        }
        try {
            const constraints = {
                video: {
                    width: { ideal: this.getVideoResolution().width },
                    height: { ideal: this.getVideoResolution().height }
                }
            };
            const response = await fetch('/api/camera/start', { method: 'POST' });
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.message || 'Failed to start camera on server');
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                video.play().catch(e => console.error('Autoplay failed:', e));
            } catch (webRtcError) {
                console.warn('WebRTC failed, relying on WebSocket frames:', webRtcError);
            }
            this.cameraStatus = 'starting';
            this.updateCameraStatusUI();
            this.showNotification('Camera starting', 'success');
        } catch (error) {
            console.error('Start webcam error:', error);
            let errorMessage = 'Failed to start webcam';
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Webcam access denied. Please grant permission.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No webcam found. Please connect a camera.';
            } else if (error.message) {
                errorMessage = error.message;
            }
            this.showNotification(errorMessage, 'error');
            this.cameraStatus = 'offline';
            this.updateCameraStatusUI();
        }
    }
    async stopWebcam() {
        const video = document.getElementById('webcam');
        if (video && video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        if (this.cameraStatus !== 'offline') {
            await fetch('/api/camera/stop', { method: 'POST' });
            this.cameraStatus = 'offline';
            this.updateCameraStatusUI();
            this.showNotification('Camera stopped', 'success');
            if (this.isRecording) {
                this.toggleRecording();
            }
        }
    }
    captureSnapshot() {
        const video = document.getElementById('webcam');
        if (!video || !video.srcObject) {
            this.showNotification('No video stream available for snapshot', 'error');
            return;
        }
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/jpeg');
        link.download = `snapshot_${new Date().toISOString()}.jpg`;
        link.click();
        this.showNotification('Snapshot captured', 'success');
    }
    toggleRecording() {
        const video = document.getElementById('webcam');
        const indicator = document.getElementById('recordingIndicator');
        const toggleButton = document.getElementById('toggleRecording');
        if (!video || !toggleButton || !indicator) {
            this.showNotification('UI elements missing for recording', 'error');
            return;
        }
        if (!this.isRecording) {
            if (!video.srcObject && this.cameraStatus !== 'online') {
                this.showNotification('Start camera before recording', 'error');
                return;
            }
            const stream = video.srcObject || (this.cameraStatus === 'online' ? { getTracks: () => [] } : null);
            if (!stream) {
                this.showNotification('No video stream available for recording', 'error');
                return;
            }
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
            this.recordedChunks = [];
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.recordedChunks.push(e.data);
            };
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `recording_${new Date().toISOString()}.webm`;
                link.click();
                URL.revokeObjectURL(url);
                this.showNotification('Recording saved', 'success');
            };
            this.mediaRecorder.start();
            this.isRecording = true;
            indicator.classList.remove('hidden');
            toggleButton.innerHTML = '<i class="fas fa-stop-circle mr-2"></i> Stop Recording';
            toggleButton.classList.remove('bg-purple-600', 'hover:bg-purple-700');
            toggleButton.classList.add('bg-red-600', 'hover:bg-red-700');
        } else {
            this.mediaRecorder.stop();
            this.isRecording = false;
            indicator.classList.add('hidden');
            toggleButton.innerHTML = '<i class="fas fa-record-vinyl mr-2"></i> Record';
            toggleButton.classList.remove('bg-red-600', 'hover:bg-red-700');
            toggleButton.classList.add('bg-purple-600', 'hover:bg-purple-700');
        }
    }
    getVideoResolution() {
        const quality = document.getElementById('videoQuality')?.value || 'medium';
        switch (quality) {
            case 'low': return { width: 640, height: 480 };
            case 'medium': return { width: 1280, height: 720 };
            case 'high': return { width: 1920, height: 1080 };
            default: return { width: 1280, height: 720 };
        }
    }
    renderDetections(data) {
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('detectionCanvas');
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        canvas.width = data.width || video.videoWidth || 1280;
        canvas.height = data.height || video.videoHeight || 720;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        data.detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            ctx.strokeStyle = det.label === 'face' ? 'blue' : 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = det.label === 'face' ? 'blue' : 'red';
            ctx.font = '14px Arial';
            ctx.fillText(`${det.label} (${(det.confidence * 100).toFixed(1)}%)`, x1, y1 - 5);
        });
    }
    updateCameraStatusUI() {
        const cameraList = document.getElementById('cameraList');
        if (cameraList) {
            cameraList.innerHTML = this.cameras.map(camera => `
                <div class="bg-dark-700 rounded-lg p-4 border border-dark-600">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-4">
                            <div class="w-12 h-12 bg-dark-600 rounded-lg flex items-center justify-center">
                                <i class="fas fa-video text-gray-400"></i>
                            </div>
                            <div>
                                <h4 class="font-medium text-white">${camera.name}</h4>
                                <p class="text-gray-400 text-sm">${camera.location}</p>
                                <p class="text-gray-500 text-xs">Last seen: ${this.formatDateTime(camera.last_seen)}</p>
                            </div>
                        </div>
                        <div class="flex items-center space-x-3">
                            <span class="px-3 py-1 text-xs rounded-full ${this.cameraStatus === 'online' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}">
                                ${this.cameraStatus.toUpperCase()}
                            </span>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    }
    setupNavigation() {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('data-section');
                this.showSection(section);
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                if (window.innerWidth < 1024) {
                    document.getElementById('sidebar')?.classList.add('-translate-x-full');
                    document.getElementById('overlay')?.classList.add('hidden');
                }
            });
        });
    }
    showSection(sectionName) {
        document.querySelectorAll('.section').forEach(section => section.classList.remove('active'));
        const targetSection = document.getElementById(sectionName);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionName;
            if (sectionName === 'settings') this.loadSettings();
        }
    }
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadCameras(),
                this.loadAlerts(),
                this.loadSystemStatus(),
                this.loadSettings()
            ]);
        } catch (error) {
            this.showNotification('Error loading data', 'error');
        }
    }
    async loadCameras() {
        const response = await fetch('/api/cameras');
        const data = await response.json();
        this.cameras = data.cameras;
        this.updateCameraStats(data);
        this.renderCameraList();
        this.updateCameraStatusUI();
    }
    async loadAlerts() {
        const response = await fetch('/api/alerts');
        const data = await response.json();
        this.alerts = data.alerts;
        this.updateAlertStats(data);
        this.renderRecentAlerts();
        this.renderAlertsList();
    }
    async loadSystemStatus() {
        const response = await fetch('/api/system/status');
        const data = await response.json();
        this.updateSystemStatus(data);
    }
    async loadSettings() {
        const response = await fetch('/api/settings');
        const data = await response.json();
        this.settings = data;
        this.updateSettingsForm();
    }
    updateCameraStats(data) {
        document.getElementById('totalCameras')?.textContent = data.total || 0;
        document.getElementById('onlineCameras')?.textContent = data.online || 0;
    }
    updateAlertStats(data) {
        document.getElementById('activeAlerts')?.textContent = data.unacknowledged || 0;
        document.getElementById('notificationCount')?.textContent = data.unacknowledged || 0;
    }
    updateSystemStatus(data) {
        document.getElementById('systemStatus')?.textContent = data.system_status === 'operational' ? 'Online' : 'Offline';
        document.getElementById('storageUsed')?.textContent = `${data.storage_used || 0}%`;
    }
    renderCameraList() {
        const list = document.getElementById('cameraList');
        if (list) list.innerHTML = this.cameras.map(camera => `
            <div class="bg-dark-700 rounded-lg p-4 border border-dark-600">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-4">
                        <div class="w-12 h-12 bg-dark-600 rounded-lg flex items-center justify-center">
                            <i class="fas fa-video text-gray-400"></i>
                        </div>
                        <div>
                            <h4 class="font-medium text-white">${camera.name}</h4>
                            <p class="text-gray-400 text-sm">${camera.location}</p>
                            <p class="text-gray-500 text-xs">Last seen: ${this.formatDateTime(camera.last_seen)}</p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-3">
                        <span class="px-3 py-1 text-xs rounded-full ${this.cameraStatus === 'online' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}">
                            ${this.cameraStatus.toUpperCase()}
                        </span>
                    </div>
                </div>
            </div>
        `).join('');
    }
    renderRecentAlerts() {
        const container = document.getElementById('recentAlerts');
        if (!container) return;
        const recentAlerts = this.alerts.slice(0, 5);
        if (recentAlerts.length === 0) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-shield-alt text-gray-500 text-3xl mb-3"></i>
                    <p class="text-gray-400">No recent alerts</p>
                </div>
            `;
            return;
        }
        container.innerHTML = recentAlerts.map(alert => `
            <div class="bg-dark-700 rounded-lg p-4 border border-dark-600">
                <div class="flex items-start justify-between">
                    <div class="flex items-start space-x-3">
                        <div class="p-2 rounded-lg ${this.getAlertColor(alert.severity)}">
                            <i class="fas ${this.getAlertIcon(alert.type)} text-sm"></i>
                        </div>
                        <div class="flex-1">
                            <h4 class="font-medium text-white text-sm">${alert.message}</h4>
                            <p class="text-gray-400 text-xs">${alert.camera_name}</p>
                            <p class="text-gray-500 text-xs">${this.formatDateTime(alert.timestamp)}</p>
                        </div>
                    </div>
                    <span class="px-2 py-1 text-xs rounded-full ${this.getSeverityColor(alert.severity)}">
                        ${alert.severity.toUpperCase()}
                    </span>
                </div>
            </div>
        `).join('');
    }
    renderAlertsList() {
        const container = document.getElementById('alertsList');
        if (!container) return;
        if (this.alerts.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-shield-alt text-gray-500 text-4xl mb-4"></i>
                    <p class="text-gray-400 text-lg">No alerts found</p>
                </div>
            `;
            return;
        }
        container.innerHTML = this.alerts.map(alert => `
            <div class="bg-dark-700 rounded-lg p-4 border border-dark-600 ${alert.acknowledged ? 'opacity-60' : ''}">
                <div class="flex items-start justify-between">
                    <div class="flex items-start space-x-4">
                        <div class="p-3 rounded-lg ${this.getAlertColor(alert.severity)}">
                            <i class="fas ${this.getAlertIcon(alert.type)}"></i>
                        </div>
                        <div class="flex-1">
                            <h4 class="font-medium text-white">${alert.message}</h4>
                            <p class="text-gray-400 text-sm">${alert.camera_name} • ${alert.type.replace('_', ' ')}</p>
                            <p class="text-gray-500 text-sm">${this.formatDateTime(alert.timestamp)}</p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-3">
                        <span class="px-3 py-1 text-xs rounded-full ${this.getSeverityColor(alert.severity)}">
                            ${alert.severity.toUpperCase()}
                        </span>
                        ${!alert.acknowledged ? `
                            <button onclick="app.acknowledgeAlert('${alert.id}')" class="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors">
                                Acknowledge
                            </button>
                        ` : `
                            <span class="px-3 py-1 bg-gray-600 text-gray-300 text-xs rounded">
                                Acknowledged
                            </span>
                        `}
                    </div>
                </div>
            </div>
        `).join('');
    }
    filterAlerts(filterId) {
        let filteredAlerts = [...this.alerts];
        if (filterId === 'filterUnacknowledged') {
            filteredAlerts = this.alerts.filter(a => !a.acknowledged);
        } else if (filterId === 'filterHighPriority') {
            filteredAlerts = this.alerts.filter(a => a.severity === 'high');
        }
        document.querySelectorAll('#alerts .flex button').forEach(btn => btn.classList.remove('active'));
        document.getElementById(filterId)?.classList.add('active');
        this.alerts = filteredAlerts;
        this.renderAlertsList();
    }
    async clearAlerts() {
        this.alerts = [];
        this.renderRecentAlerts();
        this.renderAlertsList();
        this.updateAlertStats({ unacknowledged: 0 });
        this.showNotification('Alerts cleared', 'success');
    }
    updateSettingsForm() {
        document.getElementById('privacyMode')?.checked = this.settings.privacy_mode || false;
        document.getElementById('faceBlurring')?.checked = this.settings.face_blurring || false;
        document.getElementById('notificationEnabled')?.checked = this.settings.notification_enabled || false;
        document.getElementById('alertSensitivity')?.value = this.settings.alert_sensitivity || 'medium';
        document.getElementById('motionThreshold')?.value = this.settings.motion_threshold || 50;
        document.getElementById('motionThresholdValue')?.textContent = this.settings.motion_threshold || 50;
        document.getElementById('yoloConfidence')?.value = this.settings.yolo_confidence || 0.5;
        document.getElementById('yoloConfidenceValue')?.textContent = this.settings.yolo_confidence || 0.5;
        document.getElementById('videoQuality')?.value = this.settings.video_quality || 'medium';
    }
    updateSettingsUI() {
        const newSettings = {
            privacy_mode: document.getElementById('privacyMode')?.checked || false,
            face_blurring: document.getElementById('faceBlurring')?.checked || false,
            notification_enabled: document.getElementById('notificationEnabled')?.checked || false,
            alert_sensitivity: document.getElementById('alertSensitivity')?.value || 'medium',
            motion_threshold: parseInt(document.getElementById('motionThreshold')?.value) || 50,
            yolo_confidence: parseFloat(document.getElementById('yoloConfidence')?.value) || 0.5,
            video_quality: document.getElementById('videoQuality')?.value || 'medium'
        };
        this.settings = { ...this.settings, ...newSettings };
    }
    async saveSettings() {
        try {
            this.updateSettingsUI();
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.settings)
            });
            const data = await response.json();
            if (data.success) {
                this.showNotification('Settings saved', 'success');
            } else {
                throw new Error(data.error || 'Failed to save settings');
            }
        } catch (error) {
            this.showNotification('Error saving settings', 'error');
        }
    }
    async acknowledgeAlert(alertId) {
        try {
            const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                this.showNotification('Alert acknowledged', 'success');
            } else {
                throw new Error(data.error || 'Failed to acknowledge alert');
            }
        } catch (error) {
            this.showNotification('Error acknowledging alert', 'error');
        }
    }
    startAutoRefresh() {
        this.updateInterval = setInterval(() => {
            this.loadCameras();
            this.loadAlerts();
            this.loadSystemStatus();
        }, 30000);
    }
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 z-50 px-4 py-3 rounded-lg text-white font-medium transition-all duration-300 transform translate-x-full ${
            type === 'success' ? 'bg-green-600' : 
            type === 'error' ? 'bg-red-600' : 
            'bg-blue-600'
        }`;
        notification.textContent = message;
        document.body.appendChild(notification);
        setTimeout(() => notification.classList.remove('translate-x-full'), 100);
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 3000);
    }
    getAlertColor(severity) {
        switch (severity) {
            case 'high': return 'bg-red-500/20 text-red-400';
            case 'medium': return 'bg-yellow-500/20 text-yellow-400';
            case 'low': return 'bg-blue-500/20 text-blue-400';
            default: return 'bg-gray-500/20 text-gray-400';
        }
    }
    getSeverityColor(severity) {
        switch (severity) {
            case 'high': return 'bg-red-500 text-white';
            case 'medium': return 'bg-yellow-500 text-black';
            case 'low': return 'bg-blue-500 text-white';
            default: return 'bg-gray-500 text-white';
        }
    }
    getAlertIcon(type) {
        switch (type) {
            case 'person_detected': return 'fa-user';
            case 'car_detected': return 'fa-car';
            case 'face_detected': return 'fa-user';
            default: return 'fa-exclamation';
        }
    }
    formatDateTime(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;
        return date.toLocaleDateString();
    }
}
const app = new WebSyncApp();
window.addEventListener('beforeunload', () => {
    app.stopAutoRefresh();
    app.stopWebcam();
});