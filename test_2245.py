import cv2
import mediapipe as mp
import math
import time
import numpy as np

class HandTracker:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplexity = modelComplexity
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(image, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
        return lmList

def calculate_angle(A, B, C):
    BA = [A[0] - B[0], A[1] - B[1]]
    BC = [C[0] - B[0], C[1] - B[1]]
    dot_product = BA[0]*BC[0] + BA[1]*BC[1]
    mag_BA = math.sqrt(BA[0]**2 + BA[1]**2)
    mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    if mag_BA * mag_BC == 0:
        return 0
    cosine_angle = dot_product / (mag_BA * mag_BC)
    cosine_angle = max(min(cosine_angle, 1), -1)
    angle = math.acos(cosine_angle)
    return math.degrees(angle)

# Smoothing function for angle measurements
def smooth_angles(angle_list, window_size=3):
    if len(angle_list) < window_size:
        return angle_list[-1] if angle_list else 0
    return np.mean(angle_list[-window_size:])

# Calibration system
calibration_data = {"Index": [], "Middle": [], "Ring": [], "Pinky": []}
baseline_angles = {"Index": None, "Middle": None, "Ring": None, "Pinky": None}
calibration_complete = False
calibration_start_time = 0
CALIBRATION_DURATION = 3  # seconds

def update_baseline():
    global baseline_angles, calibration_data
    for finger in calibration_data:
        if calibration_data[finger]:
            baseline_angles[finger] = np.mean(calibration_data[finger])

angle_history = {finger: [] for finger in ["Index", "Middle", "Ring", "Pinky"]}
SMOOTHING_WINDOW = 5
HISTORY_LENGTH = 10

def detect_anomaly(lmList, movement_threshold=15, straight_threshold=10):
    global calibration_complete, baseline_angles
    
    anomalies = []
    current_angles = {}
    
    fingers = {
        "Index": [5, 6, 7],
        "Middle": [9, 10, 11],
        "Ring": [13, 14, 15],
        "Pinky": [17, 18, 19]
    }
    
    # Calculate current angles
    for finger, ids in fingers.items():
        if len(lmList) > max(ids):
            A = (lmList[ids[0]][1], lmList[ids[0]][2])
            B = (lmList[ids[1]][1], lmList[ids[1]][2])
            C = (lmList[ids[2]][1], lmList[ids[2]][2])
            angle = calculate_angle(A, B, C)
            current_angles[finger] = angle
            
            # Update history
            angle_history[finger].append(angle)
            if len(angle_history[finger]) > HISTORY_LENGTH:
                angle_history[finger].pop(0)
    
    # During calibration phase
    if not calibration_complete:
        for finger in current_angles:
            calibration_data[finger].append(current_angles[finger])
        return []
    
    # Anomaly detection logic
    for finger in current_angles:
        if baseline_angles[finger] is None or len(angle_history[finger]) < 2:
            continue
            
        # Apply smoothing
        smoothed_angle = smooth_angles(angle_history[finger], SMOOTHING_WINDOW)
        
        # Calculate movement range
        angle_range = max(angle_history[finger]) - min(angle_history[finger])
        
        # Calculate relative straightness (how much straighter than baseline)
        relative_straightness = smoothed_angle - baseline_angles[finger]
        
        # Detect anomaly: significant movement + unusually straight finger
        if (angle_range > movement_threshold and 
            relative_straightness > straight_threshold):
            anomalies.append((finger, int(smoothed_angle)))
    
    return anomalies

def main():
    global calibration_complete, calibration_start_time
    
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    calibration_start_time = time.time()
    
    while True:
        success, image = cap.read()
        if not success:
            break
            
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image, draw=True)
        
        # Calibration phase
        if not calibration_complete:
            elapsed = time.time() - calibration_start_time
            remaining = max(0, CALIBRATION_DURATION - elapsed)
            
            cv2.putText(image, f"Calibrating: {remaining:.1f}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if elapsed >= CALIBRATION_DURATION:
                calibration_complete = True
                update_baseline()
        
        # Detection phase
        if lmList and calibration_complete:
            anomalies = detect_anomaly(lmList)
            
            if anomalies:
                y_offset = 60
                for finger, angle in anomalies:
                    msg = f"{finger} Anomaly ({angle}°)"
                    cv2.putText(image, msg, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += 30
            else:
                cv2.putText(image, "All Fingers Normal", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Finger Anomaly Detection", image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()