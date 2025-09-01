import cv2
import mediapipe as mp
import math
import time
import numpy as np

class HandTracker:
    def _init_(self, mode=False, max_hands=1, detection_con=0.5, model_complexity=1, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.model_complexity = model_complexity
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.landmark_drawing_spec,
                        self.connection_drawing_spec
                    )
        return image

    def get_position(self, image, hand_no=0, draw=True):
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                for idx, landmark in enumerate(hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    lm_list.append([idx, cx, cy])
                    if draw and idx in [4, 8, 12, 16, 20]:  # Draw fingertips
                        cv2.circle(image, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
        return lm_list

def calculate_angle(A, B, C):
    """Calculate angle at point B formed by points A, B, C"""
    if not (isinstance(A, tuple) and isinstance(B, tuple) and isinstance(C, tuple)):
        raise ValueError("Points must be tuples (x, y)")
    
    BA = [A[0] - B[0], A[1] - B[1]]
    BC = [C[0] - B[0], C[1] - B[1]]
    
    dot_product = BA[0]*BC[0] + BA[1]*BC[1]
    mag_BA = math.sqrt(BA[0]*2 + BA[1]*2)
    mag_BC = math.sqrt(BC[0]*2 + BC[1]*2)
    
    if mag_BA * mag_BC < 1e-5:  # Avoid division by zero
        return 0.0
        
    cosine_angle = dot_product / (mag_BA * mag_BC)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)  # Clamp to valid range
    
    return math.degrees(math.acos(cosine_angle))

class AnomalyDetector:
    FINGER_CONFIG = {
        "Thumb": {"base": 2, "joint": 3, "tip": 4},  # Landmark indices
        "Index": {"base": 5, "joint": 6, "tip": 7},
        "Middle": {"base": 9, "joint": 10, "tip": 11},
        "Ring": {"base": 13, "joint": 14, "tip": 15},
        "Pinky": {"base": 17, "joint": 18, "tip": 19}
    }
    
    def _init_(self, history_length=15):
        self.angle_history = {finger: [] for finger in self.FINGER_CONFIG}
        self.calibration_data = {
            "open_hand": {finger: None for finger in self.FINGER_CONFIG},
            "closed_fist": {finger: None for finger in self.FINGER_CONFIG}
        }
        self.is_calibrated = False
        self.history_length = history_length

    def update_history(self, finger, angle):
        """Maintain a sliding window of angle measurements"""
        self.angle_history[finger].append(angle)
        if len(self.angle_history[finger]) > self.history_length:
            self.angle_history[finger].pop(0)

    def calibrate(self, lm_list, pose_type):
        """Record finger angles for calibration poses"""
        for finger, landmarks in self.FINGER_CONFIG.items():
            if len(lm_list) > max(landmarks.values()):
                A = (lm_list[landmarks["base"]][1], lm_list[landmarks["base"]][2])
                B = (lm_list[landmarks["joint"]][1], lm_list[landmarks["joint"]][2])
                C = (lm_list[landmarks["tip"]][1], lm_list[landmarks["tip"]][2])
                angle = calculate_angle(A, B, C)
                
                if self.calibration_data[pose_type][finger] is None:
                    self.calibration_data[pose_type][finger] = []
                self.calibration_data[pose_type][finger].append(angle)

    def finalize_calibration(self):
        """Calculate average angles from calibration data"""
        for pose_type in self.calibration_data:
            for finger in self.calibration_data[pose_type]:
                if self.calibration_data[pose_type][finger]:
                    self.calibration_data[pose_type][finger] = np.mean(
                        self.calibration_data[pose_type][finger]
                    )
        self.is_calibrated = True
        print("Calibration Complete:")
        print(f"Open Hand: {self.calibration_data['open_hand']}")
        print(f"Closed Fist: {self.calibration_data['closed_fist']}")

    def detect_anomaly(self, lm_list, movement_threshold=15, straight_threshold=25):
        """Detect finger anomalies based on calibrated norms"""
        anomalies = []
        if not self.is_calibrated or not lm_list:
            return anomalies

        for finger, landmarks in self.FINGER_CONFIG.items():
            if len(lm_list) <= max(landmarks.values()):
                continue
                
            A = (lm_list[landmarks["base"]][1], lm_list[landmarks["base"]][2])
            B = (lm_list[landmarks["joint"]][1], lm_list[landmarks["joint"]][2])
            C = (lm_list[landmarks["tip"]][1], lm_list[landmarks["tip"]][2])
            current_angle = calculate_angle(A, B, C)
            self.update_history(finger, current_angle)
            
            # Skip if insufficient history
            if len(self.angle_history[finger]) < 5:
                continue
                
            # Calculate movement metrics
            angle_range = np.ptp(self.angle_history[finger])
            smoothed_angle = np.mean(self.angle_history[finger][-5:])
            open_avg = self.calibration_data["open_hand"][finger]
            
            # Anomaly detection logic
            if angle_range > movement_threshold:
                relative_straightness = smoothed_angle - open_avg
                if relative_straightness > straight_threshold:
                    anomalies.append((finger, int(smoothed_angle)))
                    
        return anomalies

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    tracker = HandTracker()
    detector = AnomalyDetector()
    
    # Calibration state management
    CALIBRATION_FRAMES = 60
    calibration_stage = "open_hand"
    calibration_count = 0
    calibration_complete = False
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame)
        lm_list = tracker.get_position(frame)
        
        # Calibration process
        if not calibration_complete:
            if calibration_stage == "open_hand":
                cv2.putText(frame, "CALIBRATION: Show OPEN HAND", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Progress: {calibration_count}/{CALIBRATION_FRAMES}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if lm_list:
                    detector.calibrate(lm_list, "open_hand")
                    calibration_count += 1
                    
                if calibration_count >= CALIBRATION_FRAMES:
                    calibration_stage = "closed_fist"
                    calibration_count = 0
                    
            elif calibration_stage == "closed_fist":
                cv2.putText(frame, "CALIBRATION: Make a FIST", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Progress: {calibration_count}/{CALIBRATION_FRAMES}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if lm_list:
                    detector.calibrate(lm_list, "closed_fist")
                    calibration_count += 1
                    
                if calibration_count >= CALIBRATION_FRAMES:
                    detector.finalize_calibration()
                    calibration_complete = True
        
        # Anomaly detection
        elif lm_list:
            anomalies = detector.detect_anomaly(lm_list)
            
            if anomalies:
                y_offset = 60
                for finger, angle in anomalies:
                    cv2.putText(frame, f"{finger} ANOMALY: {angle}°", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += 30
            else:
                cv2.putText(frame, "NORMAL OPERATION", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Glove Anomaly Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
yeh wala perplexity ne diya hai gemini ke code ko modify karke diye hai dekh ho yeh bhi