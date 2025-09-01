import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import time
import os
import csv
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    def __init__(self, fingers, window_size=200, threshold=2.5, min_std=8.0, confirmation_frames=5):
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
            if len(self.angle_history[finger]) < 50:
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

def create_dashboard(angle_dict, anomaly_dict, fps, frame_width, arduino_angles=None):
    dashboard = np.zeros((180, frame_width, 3), dtype=np.uint8)
    cv2.putText(dashboard, f"FPS: {fps}", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    max_angle = 180
    bar_width = 40
    spacing = 15
    start_x = 20
    fingers = list(angle_dict.keys())
    for idx, finger in enumerate(fingers):
        x = start_x + idx*(bar_width + spacing)
        angle = angle_dict[finger]
        bar_height = int((angle / max_angle) * 100)
        color = (0, 0, 255) if anomaly_dict.get(finger, False) else (0, 255, 0)
        cv2.rectangle(dashboard, (x, 150), (x + bar_width, 150 - bar_height), color, -1)
        cv2.putText(dashboard, finger[:3], (x + 5, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(dashboard, f"{int(angle)}°", (x + 5, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Flex sensor value display (always a number, default 0)
        if arduino_angles is not None and finger != "Thumb":
            flex_idx = idx - 1
            flex_val = arduino_angles[flex_idx] if (flex_idx < len(arduino_angles) and arduino_angles[flex_idx] is not None) else 0
            cv2.putText(dashboard, f"Flex:{flex_val}°", (x, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return dashboard

def main():
    user_id = input("Enter user name or ID: ")
    trial_id = input("Enter trial number: ")
    csv_filename = f"smartglove_{user_id}_trial{trial_id}.csv"
    csv_fields = ["timestamp", "user", "trial", "Thumb", "Index", "Middle", "Ring", "Pinky",
                  "Flex1", "Flex2", "Flex3", "Flex4", "Anom_Thumb", "Anom_Index", "Anom_Middle", "Anom_Ring", "Anom_Pinky"]
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)

    # --- Insert your Arduino serial reading code here ---
    # For example, use a thread or polling to update arduino_angles
    # For now, we'll use dummy values
    arduino_angles = [0, 0, 0, 0]  # Replace with your real Arduino reading logic

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
    anomaly_detector = EnhancedAnomalyDetector(fingers, window_size=200, threshold=2.5, min_std=8.0, confirmation_frames=5)
    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    angle_history = {finger: deque([0]*200, maxlen=200) for finger in fingers}
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

        # --- Arduino flex sensor readings ---
        # arduino_angles = ... (update this from your Arduino code)

        # --- ThingSpeak upload ---
        now = time.time()
        if not hasattr(main, "last_thingspeak_time"):
            main.last_thingspeak_time = 0
        if now - main.last_thingspeak_time > THINGSPEAK_UPDATE_INTERVAL:
            send_to_thingspeak(angle_dict, arduino_angles)
            main.last_thingspeak_time = now

        # --- CSV logging for this user/trial ---
        timestamp = time.time()
        row = [
            timestamp, user_id, trial_id,
            angle_dict["Thumb"], angle_dict["Index"], angle_dict["Middle"], angle_dict["Ring"], angle_dict["Pinky"],
            arduino_angles[0] if len(arduino_angles) > 0 else "",
            arduino_angles[1] if len(arduino_angles) > 1 else "",
            arduino_angles[2] if len(arduino_angles) > 2 else "",
            arduino_angles[3] if len(arduino_angles) > 3 else "",
            anomalies["Thumb"], anomalies["Index"], anomalies["Middle"], anomalies["Ring"], anomalies["Pinky"]
        ]
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # --- Visualization ---
        dashboard = create_dashboard(angle_dict, anomalies, fps, 640, arduino_angles=arduino_angles)
        webcam = cv2.resize(frame, (320, 300))
        graph = np.zeros((300, 640, 3), dtype=np.uint8)  # Placeholder for your dynamic graph
        top_row = np.hstack([webcam, graph])
        combined = np.vstack([top_row, dashboard])
        cv2.putText(combined, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking + Flex Sensor Verification", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()