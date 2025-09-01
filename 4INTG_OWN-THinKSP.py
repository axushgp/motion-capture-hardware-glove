## WORKS !!!! *THIS*
## BEST FINAL
# minus dashboard

import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import time
import serial
import threading
import os
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ThingSpeak setup
THINGSPEAK_API_KEY = "U9XSRQT10HU4VED1"  # <-- Replace with your ThingSpeak Write API Key
THINGSPEAK_UPDATE_INTERVAL = 20  # seconds

def send_to_thingspeak(angles, flex_angles):
    url = "https://api.thingspeak.com/update"
    data = {
        "api_key": THINGSPEAK_API_KEY,
        "field1": angles.get("Thumb", 0),
        "field2": angles.get("Index", 0),
        "field3": angles.get("Middle", 0),
        "field4": angles.get("Ring", 0),
        "field5": angles.get("Pinky", 0),
        "field6": flex_angles[0] if len(flex_angles) > 0 and flex_angles[0] is not None else 0,
        "field7": flex_angles[1] if len(flex_angles) > 1 and flex_angles[1] is not None else 0,
        "field8": flex_angles[2] if len(flex_angles) > 2 and flex_angles[2] is not None else 0,
    }
    try:
        response = requests.get(url, params=data, timeout=5)
        print(f"[ThingSpeak] Response: {response.status_code}, Data sent: {data}")
    except Exception as e:
        print(f"[ThingSpeak] Error sending data: {e}")

class ArduinoReader(threading.Thread):
    def __init__(self, port='COM4', baudrate=9600):
        super().__init__()
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.angles = [None, None, None, None]
        self.temp_angles = []
        self.daemon = True
        self.start()
    def run(self):
        while True:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    print(f"[SERIAL] {line}")
                    if "Angle" in line:
                        parts = line.split("Angle :")
                        if len(parts) == 2:
                            angle_str = parts[1].replace("°", "").strip()
                            angle = int(angle_str)
                            self.temp_angles.append(angle)
                            if len(self.temp_angles) == 4:
                                self.angles = self.temp_angles.copy()
                                self.temp_angles = []
            except Exception as e:
                print(f"[ERROR] {e}")
    def get_angles(self):
        return self.angles.copy()
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
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
    if not all(isinstance(point, (tuple, list)) and len(point) == 2 for point in [A, B, C]):
        return 0.0
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    mag_BA = math.sqrt(BA[0]**2 + BA[1]**2)
    mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    if mag_BA == 0 or mag_BC == 0:
        return 0.0
    cosine_angle = dot_product / (mag_BA * mag_BC)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)
    return math.degrees(math.acos(cosine_angle))

class EnhancedAnomalyDetector:
    def __init__(self, fingers, window_size=60, threshold=2.5, min_std=8.0, confirmation_frames=3):
        self.window_size = window_size
        self.threshold = threshold
        self.min_std = min_std
        self.confirmation_frames = confirmation_frames
        self.angle_history = {finger: deque(maxlen=window_size) for finger in fingers}
        self.anomaly_counters = {finger: 0 for finger in fingers}
        self.confirmed_anomalies = {finger: False for finger in fingers}
        self.expected_ranges = {
            "Thumb": (0, 30, 150, 180),
            "Index": (0, 20, 160, 180),
            "Middle": (0, 20, 160, 180),
            "Ring": (0, 20, 160, 180),
            "Pinky": (0, 20, 160, 180)
        }
    def is_half_folded(self, finger, angle):
        low_min, low_max, high_min, high_max = self.expected_ranges[finger]
        return not (low_min <= angle <= low_max or high_min <= angle <= high_max)
    def update(self, angle_dict):
        anomalies = {}
        for finger, angle in angle_dict.items():
            self.angle_history[finger].append(angle)
            if len(self.angle_history[finger]) < 15:
                anomalies[finger] = False
                continue
            is_half_folded = self.is_half_folded(finger, angle)
            history = list(self.angle_history[finger])
            mean = np.mean(history)
            std = np.std(history)
            if std < self.min_std:
                anomalies[finger] = is_half_folded
                self.anomaly_counters[finger] = 0
                continue
            z_score = abs(angle - mean) / std
            is_stat_anomaly = z_score > self.threshold
            is_anomaly = is_stat_anomaly or is_half_folded
            if is_anomaly:
                self.anomaly_counters[finger] += 1
                if self.anomaly_counters[finger] >= self.confirmation_frames:
                    anomalies[finger] = True
                    self.confirmed_anomalies[finger] = True
                else:
                    anomalies[finger] = False
            else:
                self.anomaly_counters[finger] = 0
                anomalies[finger] = self.confirmed_anomalies[finger]
        return anomalies

def create_dashboard(angle_dict, arduino_angles, anomaly_dict, arduino_anomaly, fps, frame_width):
    dashboard = np.zeros((220, frame_width, 3), dtype=np.uint8)
    cv2.putText(dashboard, f"FPS: {fps}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    max_angle = 180
    bar_width = 40
    spacing = 20
    start_x = 20
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for idx, finger in enumerate(fingers):
        x = start_x + idx * (bar_width + spacing)
        angle = angle_dict.get(finger, 0)
        bar_height = int((angle / max_angle) * 100)
        color = (0, 0, 255) if anomaly_dict.get(finger, False) else (0, 255, 0)
        cv2.rectangle(dashboard, (x, 150), (x + bar_width, 150 - bar_height), color, -1)
        cv2.putText(dashboard, finger[:3], (x + 5, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(dashboard, f"{int(angle)}°", (x + 5, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if finger != "Thumb":
            flex_idx = idx - 1
            if flex_idx < len(arduino_angles) and arduino_angles[flex_idx] is not None:
                flex_val = arduino_angles[flex_idx]
                cv2.putText(dashboard, f"Flex:{flex_val}°", (x, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(dashboard, "Flex:N/C", (x, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return dashboard

def create_dynamic_graph(angle_history, graph_width, graph_height, anomalies, current_frame):
    graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    for y in range(0, 181, 30):
        y_pos = graph_height - int(y/180 * graph_height)
        color = (50, 50, 50) if y % 60 != 0 else (100, 100, 100)
        cv2.line(graph, (0, y_pos), (graph_width, y_pos), color, 1)
        cv2.putText(graph, f"{y}°", (5, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    colors = [(0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    max_history = max(len(h) for h in angle_history.values())
    for idx, (finger, history) in enumerate(angle_history.items()):
        color = colors[idx]
        for i in range(1, len(history)):
            x1 = int((i-1) / max_history * graph_width)
            y1 = graph_height - int(history[i-1]/180 * graph_height)
            x2 = int(i / max_history * graph_width)
            y2 = graph_height - int(history[i]/180 * graph_height)
            line_thickness = 3 if abs(history[i] - history[i-1]) > 30 else 1
            cv2.line(graph, (x1, y1), (x2, y2), color, line_thickness)
            if i == len(history) - 1:
                cv2.circle(graph, (x2, y2), 6, color, -1)
                cv2.putText(graph, f"{int(history[i])}°",
                           (x2 + 10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    legend_x = graph_width - 150
    legend_y = 30
    for idx, finger in enumerate(angle_history.keys()):
        color = colors[idx]
        cv2.rectangle(graph, (legend_x, legend_y), (legend_x + 20, legend_y + 20), color, -1)
        cv2.putText(graph, finger, (legend_x + 30, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 30
    status = "NORMAL"
    status_color = (0, 255, 0)
    if any(anomalies.values()):
        status = "ANOMALY DETECTED"
        status_color = (0, 0, 255)
    cv2.putText(graph, f"STATUS: {status}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(graph, f"Frame: {current_frame}", (graph_width - 150, graph_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return graph

def main():
    arduino = ArduinoReader(port='COM4', baudrate=9600)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_joints = {
        "Thumb": {"A": 2, "B": 3, "C": 4},
        "Index": {"A": 5, "B": 6, "C": 8},
        "Middle": {"A": 9, "B": 10, "C": 12},
        "Ring": {"A": 13, "B": 14, "C": 16},
        "Pinky": {"A": 17, "B": 18, "C": 20}
    }
    anomaly_detector = EnhancedAnomalyDetector(fingers)
    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    angle_history = {finger: deque([0]*50, maxlen=50) for finger in fingers}
    graph_height = 300
    last_thingspeak_time = 0
    print("System started. Press ESC to exit.")
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera read error.")
            break
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
                            angle_history[finger].append(angle)
                    except Exception:
                        angle_dict[finger] = 0
                        angle_history[finger].append(0)
        for finger in fingers:
            if finger not in angle_dict:
                angle_dict[finger] = 0
                angle_history[finger].append(0)
        anomalies = anomaly_detector.update(angle_dict)
        arduino_angles = arduino.get_angles()
        arduino_anomaly = []
        for a in arduino_angles:
            if a is not None and (a < 30 or a > 150):
                arduino_anomaly.append(False)
            else:
                arduino_anomaly.append(True)
        double_anomaly = {}
        for idx, finger in enumerate(fingers):
            if finger == "Thumb":
                double_anomaly[finger] = anomalies[finger]
            else:
                flex_idx = idx - 1
                double_anomaly[finger] = anomalies[finger] and (arduino_anomaly[flex_idx] if flex_idx < len(arduino_anomaly) else False)
        graph = create_dynamic_graph(angle_history, 640, graph_height, double_anomaly, frame_count)
        webcam = cv2.resize(frame, (320, graph_height))
        top_row = np.hstack([webcam, graph])
        dashboard = create_dashboard(angle_dict, arduino_angles, anomalies, arduino_anomaly, fps, top_row.shape[1])
        if dashboard.shape[1] != top_row.shape[1]:
            dashboard = cv2.resize(dashboard, (top_row.shape[1], dashboard.shape[0]))
        combined = np.vstack([top_row, dashboard])
        cv2.imshow("Hand Tracking + Flex Sensor Verification", combined)

        # --- THINGSPEAK UPLOAD ---
        now = time.time()
        if now - last_thingspeak_time > THINGSPEAK_UPDATE_INTERVAL:
            send_to_thingspeak(angle_dict, arduino_angles)
            last_thingspeak_time = now

        if cv2.waitKey(1) & 0xFF == 27:
            break
    arduino.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()