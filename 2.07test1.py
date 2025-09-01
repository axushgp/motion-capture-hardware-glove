import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import ctypes

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except:
        return 1280, 720

def resize_to_screen(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class EnhancedHandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.8, tracking_confidence=0.8):
        self.hands = mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results
    def get_landmarks(self, image, hand_landmarks):
        h, w, _ = image.shape
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks.append((x, y))
        return landmarks

def calculate_angle(A, B, C):
    try:
        if not all(isinstance(point, (tuple, list)) and len(point) == 2 for point in [A, B, C]):
            return 0.0
        BA = (A[0] - B[0], A[1] - B[1])
        BC = (C[0] - B[0], C[1] - B[1])
        dot_product = BA[0] * BC[0] + BA[1] * BC[1]
        mag_BA = math.sqrt(BA[0]*2 + BA[1]*2)
        mag_BC = math.sqrt(BC[0]*2 + BC[1]*2)
        if mag_BA == 0 or mag_BC == 0:
            return 0.0
        cosine_angle = dot_product / (mag_BA * mag_BC)
        cosine_angle = max(min(cosine_angle, 1.0), -1.0)
        return math.degrees(math.acos(cosine_angle))
    except Exception:
        return 0.0

class ScientificAnomalyDetector:
    def __init__(self, fingers, window_size=90, threshold=3.0, min_std=5.0, confirmation_frames=5):
        self.window_size = window_size
        self.threshold = threshold
        self.min_std = min_std
        self.confirmation_frames = confirmation_frames
        self.angle_history = {finger: deque(maxlen=window_size) for finger in fingers}
        self.anomaly_counters = {finger: 0 for finger in fingers}
        self.confirmed_anomalies = {finger: False for finger in fingers}
        self.baseline_established = {finger: False for finger in fingers}
        self.baseline_values = {finger: (0, 0) for finger in fingers}
        self.expected_ranges = {
            "Thumb": (0, 180),
            "Index": (0, 180),
            "Middle": (0, 180),
            "Ring": (0, 180),
            "Pinky": (0, 180)
        }
    def establish_baseline(self, finger):
        if len(self.angle_history[finger]) == self.window_size and not self.baseline_established[finger]:
            history = list(self.angle_history[finger])
            mean = np.mean(history)
            std = np.std(history)
            self.baseline_values[finger] = (mean, max(std, self.min_std))
            self.baseline_established[finger] = True
    def is_biomechanical_anomaly(self, finger, angle):
        min_val, max_val = self.expected_ranges[finger]
        return angle < min_val or angle > max_val
    def update(self, angle_dict):
        anomalies = {}
        for finger, angle in angle_dict.items():
            self.angle_history[finger].append(angle)
            self.establish_baseline(finger)
            if not self.baseline_established[finger]:
                anomalies[finger] = False
                continue
            mean, std = self.baseline_values[finger]
            # Stricter threshold for thumb
            if finger == "Thumb":
                eff_threshold = self.threshold + 0.7
            else:
                eff_threshold = self.threshold + (0.5 if self.is_biomechanical_anomaly(finger, angle) else 0)
            z_score = abs(angle - mean) / std if std > 0 else 0
            is_stat_anomaly = z_score > eff_threshold
            is_biomech_anomaly = self.is_biomechanical_anomaly(finger, angle)
            is_anomaly = is_stat_anomaly or is_biomech_anomaly
            if is_anomaly:
                self.anomaly_counters[finger] += 1
                if self.anomaly_counters[finger] >= self.confirmation_frames:
                    anomalies[finger] = True
                    self.confirmed_anomalies[finger] = True
                else:
                    anomalies[finger] = False
            else:
                self.anomaly_counters[finger] = max(0, self.anomaly_counters[finger] - 1)
                anomalies[finger] = self.confirmed_anomalies[finger] and self.anomaly_counters[finger] > 0
        return anomalies

def create_dashboard(angle_dict, anomaly_dict, fps, analysis_results, recording_time, frame_size):
    dashboard = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    # Border
    cv2.rectangle(dashboard, (0, 0), (frame_size[0]-1, frame_size[1]-1), (0, 200, 255), 2)
    # Heading
    cv2.putText(dashboard, "ANOMALY DETECTION SYSTEM", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)
    # Status + FPS
    status = "NORMAL"
    status_color = (0, 255, 0)
    if any(anomaly_dict.values()):
        status = "ANOMALY DETECTED"
        status_color = (0, 0, 255)
    cv2.putText(dashboard, f"Status: {status}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (frame_size[0]-170, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Recording
    rec_color = (0, 255, 255) if recording_time > 0 else (180, 180, 180)
    cv2.putText(dashboard, f"Recording: {'ON' if recording_time > 0 else 'OFF'}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
    cv2.putText(dashboard, f"Time: {recording_time:.1f}s", (260, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
    # Finger angles
    y_base = 170
    for i, (finger, angle) in enumerate(angle_dict.items()):
        color = (0, 0, 255) if anomaly_dict.get(finger, False) else (0, 255, 0)
        cv2.putText(dashboard, f"{finger}: {int(angle)}", (30, y_base + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # Analysis
    cv2.putText(dashboard, "Analysis:", (30, y_base + 6*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
    for i, (finger, result) in enumerate(analysis_results.items()):
        color = (0, 255, 0) if "Normal" in result else (0, 0, 255)
        cv2.putText(dashboard, f"{finger}: {result}", (30, y_base + (7+i)*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Optional: Add your name/LinkedIn at bottom
    cv2.putText(dashboard, "by Your Name | linkedin.com/in/yourprofile", (20, frame_size[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    return dashboard

def create_scientific_graph(angle_history, graph_size, anomalies, current_frame):
    try:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(graph_size[0]/100, graph_size[1]/100), dpi=100)
        ax = fig.add_subplot(111)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        max_len = 0
        for idx, (finger, history) in enumerate(angle_history.items()):
            if len(history) > 0:
                x = np.arange(len(history))
                y = np.array(history)
                ax.plot(x, y, color=colors[idx % len(colors)], label=finger, linewidth=2, alpha=0.8)
                max_len = max(max_len, len(history))
                if anomalies.get(finger, False) and len(history) > 0:
                    ax.scatter([len(history)-1], [history[-1]], color='red', s=100, zorder=5, marker='x')
        ax.set_title("Real-Time Finger Flexion", fontsize=10, color='white')
        ax.set_xlabel("Frame", fontsize=8, color='white')
        ax.set_ylabel("Angle", fontsize=8, color='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_ylim(0, 200)
        if max_len > 0:
            ax.set_xlim(0, max(50, max_len))
        ax.tick_params(colors='white', labelsize=7)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        blank_graph = np.zeros((graph_size[1], graph_size[0], 3), dtype=np.uint8)
        cv2.putText(blank_graph, "Graph Loading...", (50, graph_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank_graph

def analyze_performance(data):
    analysis = {}
    for finger, history in data.items():
        if len(history) == 0:
            analysis[finger] = "No data"
            continue
        angles = np.array(history)
        mean_angle = np.mean(angles)
        std_dev = np.std(angles)
        z_scores = np.abs((angles - mean_angle) / (std_dev + 1e-5))
        anomaly_count = np.sum(z_scores > 3.0)
        anomaly_percentage = (anomaly_count / len(angles)) * 100
        biomech_anomalies = np.sum((angles < 0) | (angles > 180))
        if anomaly_percentage > 15 or biomech_anomalies > 0:
            analysis[finger] = f"Strain ({anomaly_percentage:.1f}%)"
        elif std_dev > 20:
            analysis[finger] = f"Variable ({std_dev:.1f}°)"
        else:
            analysis[finger] = "Normal"
    return analysis

def main():
    frame_width, frame_height = 640, 360
    panel_height = 400
    dashboard_width = 420
    screen_w, screen_h = get_screen_size()
    screen_w, screen_h = int(screen_w * 0.9), int(screen_h * 0.9)
    window_name = "Anomaly Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    tracker = EnhancedHandTracker()
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_joints = {
        "Thumb": {"A": 2, "B": 3, "C": 4},
        "Index": {"A": 5, "B": 6, "C": 8},
        "Middle": {"A": 9, "B": 10, "C": 12},
        "Ring": {"A": 13, "B": 14, "C": 16},
        "Pinky": {"A": 17, "B": 18, "C": 20}
    }
    anomaly_detector = ScientificAnomalyDetector(fingers)
    recording = False
    start_time = 0
    recorded_data = {finger: [] for finger in fingers}
    analysis_results = {finger: "Initializing" for finger in fingers}
    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    graph_size = (frame_width, panel_height)
    graph_buffer = None
    print("System ready. Controls: SPACE=Record, ESC=Exit")
    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame_count += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        results = tracker.process_frame(frame)
        angle_dict = {}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                landmarks = tracker.get_landmarks(frame, hand_landmarks)
                for finger, joints in finger_joints.items():
                    try:
                        if len(landmarks) > max(joints.values()):
                            A = landmarks[joints["A"]]
                            B = landmarks[joints["B"]]
                            C = landmarks[joints["C"]]
                            angle = calculate_angle(A, B, C)
                            angle_dict[finger] = angle
                            if recording:
                                recorded_data[finger].append(angle)
                    except Exception:
                        angle_dict[finger] = 0
        for finger in fingers:
            if finger not in angle_dict:
                angle_dict[finger] = 0
        anomalies = anomaly_detector.update(angle_dict)
        if frame_count % 3 == 0 or graph_buffer is None:
            graph_buffer = create_scientific_graph(
                anomaly_detector.angle_history,
                graph_size,
                anomalies,
                frame_count
            )
        recording_time = time.time() - start_time if recording else 0
        dashboard = create_dashboard(
            angle_dict,
            anomalies,
            fps,
            analysis_results,
            recording_time,
            (dashboard_width, panel_height)
        )
        frame = cv2.resize(frame, (frame_width, frame_height))
        graph_buffer = cv2.resize(graph_buffer, (frame_width, panel_height - frame_height))
        left_panel = np.vstack([frame, graph_buffer])
        # Make sure dashboard and left_panel have the same height
        if left_panel.shape[0] != dashboard.shape[0]:
            dashboard = cv2.resize(dashboard, (dashboard_width, left_panel.shape[0]))
        combined = np.hstack([left_panel, dashboard])
        display_frame = resize_to_screen(combined, screen_w, screen_h)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            recording = not recording
            if recording:
                start_time = time.time()
                recorded_data = {finger: [] for finger in fingers}
                analysis_results = {finger: "Recording..." for finger in fingers}
            else:
                if any(len(data) > 0 for data in recorded_data.values()):
                    analysis_results = analyze_performance(recorded_data)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")