# import cv2
# import mediapipe as mp
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import deque
# import time
# import os

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles



# class HandTracker:
#     def __init__(self, mode=False, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
#         self.hands = mp_hands.Hands(
#             static_image_mode=mode,
#             max_num_hands=max_hands,
#             min_detection_confidence=detection_confidence,
#             min_tracking_confidence=tracking_confidence
#         )

    

#     def process_frame(self, image):
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(image_rgb)
#         return results

    

#     def get_landmarks(self, image, hand_landmarks):
#         h, w, _ = image.shape
#         landmarks = []
#         for landmark in hand_landmarks.landmark:
#             x, y = int(landmark.x * w), int(landmark.y * h)
#             landmarks.append((x, y))
#         return landmarks



# def calculate_angle(A, B, C):
#     """Calculate the angle at point B between points A, B, and C"""
#     if not all(isinstance(point, (tuple, list)) and len(point) == 2 for point in [A, B, C]):
#         return 0.0

#     BA = (A[0] - B[0], A[1] - B[1])
#     BC = (C[0] - B[0], C[1] - B[1])

#     dot_product = BA[0] * BC[0] + BA[1] * BC[1]
#     mag_BA = math.sqrt(BA[0]*2 + BA[1]*2)
#     mag_BC = math.sqrt(BC[0]*2 + BC[1]*2)

#     if mag_BA == 0 or mag_BC == 0:
#         return 0.0

#     cosine_angle = dot_product / (mag_BA * mag_BC)
#     cosine_angle = max(min(cosine_angle, 1.0), -1.0)
#     return math.degrees(math.acos(cosine_angle))



# class AnomalyDetector:

#     def _init_(self, fingers, window_size=30, threshold=2.5):
#         self.window_size = window_size
#         self.threshold = threshold
#         self.angle_history = {finger: deque(maxlen=window_size) for finger in fingers}
#         self.baseline_angles = {finger: deque(maxlen=window_size) for finger in fingers}
#         self.smoothed_angles = {finger: 0 for finger in fingers}

    

#     def update(self, angle_dict):
#         anomalies = {}
#         smoothed_values = {}

        

#         for finger, angle in angle_dict.items():
#             # Add to history
#             self.angle_history[finger].append(angle)

            

#             # Calculate smoothed angle using exponential moving average

#             if self.smoothed_angles[finger] == 0:
#                 self.smoothed_angles[finger] = angle

#             else:
#                 alpha = 0.3
#                 self.smoothed_angles[finger] = alpha * angle + (1 - alpha) * self.smoothed_angles[finger]

#             smoothed_values[finger] = self.smoothed_angles[finger]

#           # Detect anomalies

#             if len(self.angle_history[finger]) >= 15:
#                 history_array = np.array(list(self.angle_history[finger]))
#                 mean_angle = np.mean(history_array)
#                 std_angle = np.std(history_array)

#                 if std_angle > 5:  # Only detect if there's sufficient variation
#                     z_score = abs(angle - mean_angle) / std_angle
#                     anomalies[finger] = z_score > self.threshold

#                 else:
#                     anomalies[finger] = False
#             else:
#                 anomalies[finger] = False

#         return anomalies, smoothed_values



# class EnhancedLivePlot:

#     def __init__(self, fingers, max_len=100):
#         self.fingers = fingers
#         self.max_len = max_len
#         self.angle_data = {finger: deque([0]*max_len, maxlen=max_len) for finger in fingers}
#         self.smoothed_data = {finger: deque([0]*max_len, maxlen=max_len) for finger in fingers}
#         self.anomaly_flags = {finger: deque([False]*max_len, maxlen=max_len) for finger in fingers}

        

#         # Initialize matplotlib

#         plt.ion()
#         self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
#         plt.subplots_adjust(hspace=0.4)

        

#         # Angle plot

#         self.angle_lines = {}
#         self.smooth_lines = {}
#         colors = ['red', 'blue', 'green', 'orange', 'purple']       

#         for I, finger in enumerate(fingers):
#             # Raw angle line
#             raw_line, = self.ax1.plot([], [], label=f'{finger} Raw', 
#                                      color=colors[i], alpha=0.7, linewidth=1.5)

#             # Smoothed angle line
#             smooth_line, = self.ax1.plot([], [], label=f'{finger} Trend', 
#                                         color=colors[i], linestyle='--', linewidth=2.5)

            

#             self.angle_lines[finger] = raw_line
#             self.smooth_lines[finger] = smooth_line

        

#         self.ax1.set_ylim(0, 180)
#         self.ax1.set_xlim(0, max_len)
#         self.ax1.set_title("Finger Joint Angles with Anomaly Detection")
#         self.ax1.set_ylabel("Angle (degrees)")
#         self.ax1.legend(loc='upper right', ncol=2, fontsize=8)
#         self.ax1.grid(True, linestyle='--', alpha=0.3)

        

#         # Anomaly indicators

#         self.anomaly_bars = []
#         for I, finger in enumerate(fingers):
#             bar = self.ax2.bar(I, 0, color='green', alpha=0.7, label=finger)
#             self.anomaly_bars.append(bar)

        

#         self.ax2.set_ylim(0, 1)
#         self.ax2.set_title("Anomaly Detection Status (Green=Normal, Red=Anomaly)")
#         self.ax2.set_xticks(range(len(fingers)))
#         self.ax2.set_xticklabels(fingers)
#         self.ax2.set_ylabel("Status")

#         # Frame counter
#         self.frame_counter = 0

#     def update(self, angle_dict, smoothed_dict, anomaly_dict):
#         self.frame_counter += 1     
#         for I, finger in enumerate(self.fingers):
#             # Update data queues
#             self.angle_data[finger].append(angle_dict.get(finger, 0))
#             self.smoothed_data[finger].append(smoothed_dict.get(finger, 0))
#             self.anomaly_flags[finger].append(anomaly_dict.get(finger, False))

            

#             # Update plot data

#             xdata = range(len(self.angle_data[finger]))
#             self.angle_lines[finger].set_data(xdata, list(self.angle_data[finger]))
#             self.smooth_lines[finger].set_data(xdata, list(self.smoothed_data[finger]))

            

#             # Update anomaly bar

#             if anomaly_dict.get(finger, False):
#                 self.anomaly_bars[I][0].set_color('red')
#                 self.anomaly_bars[I][0].set_height(1)

#             else:

#                 self.anomaly_bars[I][0].set_color('green')
#                 self.anomaly_bars[I][0].set_height(0.5)

#         # Update plot titles

#         self.ax1.set_title(f"Finger Joint Angles (Frame: {self.frame_counter})")      

#         # Refresh plot
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()



# def main():    # Initialize video capture
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open video stream.")
#         return

    

#     # Set camera resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


#     # Initialize hand tracker

#     tracker = HandTracker()


#     # Finger configuration
#     fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
#     finger_joints = {
#         "Thumb": {"A": 2, "B": 3, "C": 4},
#         "Index": {"A": 5, "B": 6, "C": 8},
#         "Middle": {"A": 9, "B": 10, "C": 12},
#         "Ring": {"A": 13, "B": 14, "C": 16},
#         "Pinky": {"A": 17, "B": 18, "C": 20}
#     } 

#     # Initialize components

#     try:
#         live_plot = EnhancedLivePlot(fingers)
#         anomaly_detector = AnomalyDetector(fingers, window_size=25, threshold=2.5)
#         print("Components initialized successfully!")

#     except Exception as e:
#         print(f"Error initializing components: {e}")
#         return

#     # FPS tracking
#     fps_counter = 0
#     fps_timer = time.time()
#     fps = 0
#     print("Hand tracking with anomaly detection started. Press ESC to exit.")

    

#     while True:
#         success, frame = cap.read()
#         if not success:
#             print("Failed to read from camera.")
#             break

#         # Calculate FPS

#         fps_counter += 1

#         if time.time() - fps_timer >= 1.0:
#             fps = fps_counter
#             fps_counter = 0
#             fps_timer = time.time()

#         # Process frame
#         results = tracker.process_frame(frame)

#         angle_dict = {}
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(
#                     frame,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )

#                 # Get landmarks
#                 landmarks = tracker.get_landmarks(frame, hand_landmarks)

#                 # Calculate angles for each finger
#                 for finger, joints in finger_joints.items():
#                     try:
#                         if len(landmarks) > max(joints.values()):
#                             A = landmarks[joints["A"]]
#                             B = landmarks[joints["B"]]
#                             C = landmarks[joints["C"]]
#                             angle = calculate_angle(A, B, C)
#                             angle_dict[finger] = angle

#                         else:
#                             angle_dict[finger] = 0
#                     except (IndexError, TypeError, KeyError) as e:
#                         angle_dict[finger] = 0

        

#         # Fill missing values
#         for finger in fingers:
#             if finger not in angle_dict:
#                 angle_dict[finger] = 0

        

#         # Detect anomalies
#         try:
#             anomalies, smoothed_angles = anomaly_detector.update(angle_dict)

#             # Update plot
#             live_plot.update(angle_dict, smoothed_angles, anomalies)

#         except Exception as e:
#             print(f"Error in anomaly detection: {e}")
#             continue

#         # Display info on frame
#         cv2.putText(frame, f"FPS: {fps}", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Display anomalies
#         anomaly_text = []
#         for finger, is_anomaly in anomalies.items():
#             if is_anomaly:
#                 anomaly_text.append(finger)
       

#         if anomaly_text:
#             cv2.putText(frame, "ANOMALY: " + ", ".join(anomaly_text), 
#                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#         else:
#             cv2.putText(frame, "STATUS: NORMAL", 
#                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # Display frame
#         cv2.imshow("Finger Joint Anomaly Detection", frame)

#         # Exit on ESC
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     plt.ioff()
#     plt.close('all')
#     print("Application closed.")

# if __name__ == "__main__":
#     main()




import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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
    """Calculate the angle at point B between points A, B, and C"""
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

class AnomalyDetector:
    def __init__(self, fingers, window_size=30, threshold=2.5):
        self.window_size = window_size
        self.threshold = threshold
        self.angle_history = {finger: deque(maxlen=window_size) for finger in fingers}
        self.smoothed_angles = {finger: 0 for finger in fingers}
    
    def update(self, angle_dict):
        anomalies = {}
        smoothed_values = {}
        
        for finger, angle in angle_dict.items():
            # Add to history
            self.angle_history[finger].append(angle)
            
            # Calculate smoothed angle using exponential moving average
            if self.smoothed_angles[finger] == 0:
                self.smoothed_angles[finger] = angle
            else:
                alpha = 0.3
                self.smoothed_angles[finger] = alpha * angle + (1 - alpha) * self.smoothed_angles[finger]
            
            smoothed_values[finger] = self.smoothed_angles[finger]
            
            # Detect anomalies
            if len(self.angle_history[finger]) >= 15:
                history_array = np.array(list(self.angle_history[finger]))
                mean_angle = np.mean(history_array)
                std_angle = np.std(history_array)
                
                if std_angle > 5:  # Only detect if there's sufficient variation
                    z_score = abs(angle - mean_angle) / std_angle
                    anomalies[finger] = z_score > self.threshold
                else:
                    anomalies[finger] = False
            else:
                anomalies[finger] = False
                
        return anomalies, smoothed_values

class EnhancedLivePlot:
    def __init__(self, fingers, max_len=100):
        self.fingers = fingers
        self.max_len = max_len
        self.angle_data = {finger: deque([0]*max_len, maxlen=max_len) for finger in fingers}
        self.smoothed_data = {finger: deque([0]*max_len, maxlen=max_len) for finger in fingers}
        
        # Initialize matplotlib
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4)
        
        # Define colors for each finger
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Angle plot
        self.angle_lines = {}
        self.smooth_lines = {}
        
        for idx, finger in enumerate(fingers):
            color = colors[idx % len(colors)]
            # Raw angle line
            raw_line, = self.ax1.plot([], [], label=f'{finger} Raw', 
                                     color=color, alpha=0.7, linewidth=1.5)
            # Smoothed angle line
            smooth_line, = self.ax1.plot([], [], label=f'{finger} Trend', 
                                        color=color, linestyle='--', linewidth=2.5)
            
            self.angle_lines[finger] = raw_line
            self.smooth_lines[finger] = smooth_line
        
        self.ax1.set_ylim(0, 180)
        self.ax1.set_xlim(0, max_len)
        self.ax1.set_title("Finger Joint Angles with Anomaly Detection")
        self.ax1.set_ylabel("Angle (degrees)")
        self.ax1.legend(loc='upper right', ncol=2, fontsize=8)
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Anomaly indicators - Fixed the variable reference issue
        self.anomaly_bars = {}
        finger_positions = list(range(len(fingers)))
        bar_colors = ['green'] * len(fingers)
        
        bars = self.ax2.bar(finger_positions, [0.5] * len(fingers), color=bar_colors, alpha=0.7)
        
        for idx, finger in enumerate(fingers):
            self.anomaly_bars[finger] = bars[idx]
        
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title("Anomaly Detection Status (Green=Normal, Red=Anomaly)")
        self.ax2.set_xticks(finger_positions)
        self.ax2.set_xticklabels(fingers)
        self.ax2.set_ylabel("Status")
        
        # Frame counter
        self.frame_counter = 0
    
    def update(self, angle_dict, smoothed_dict, anomaly_dict):
        self.frame_counter += 1
        
        for finger in self.fingers:
            # Update data queues
            self.angle_data[finger].append(angle_dict.get(finger, 0))
            self.smoothed_data[finger].append(smoothed_dict.get(finger, 0))
            
            # Update plot data
            xdata = range(len(self.angle_data[finger]))
            self.angle_lines[finger].set_data(xdata, list(self.angle_data[finger]))
            self.smooth_lines[finger].set_data(xdata, list(self.smoothed_data[finger]))
            
            # Update anomaly bar
            if anomaly_dict.get(finger, False):
                self.anomaly_bars[finger].set_color('red')
                self.anomaly_bars[finger].set_height(1)
            else:
                self.anomaly_bars[finger].set_color('green')
                self.anomaly_bars[finger].set_height(0.5)
        
        # Update plot title
        self.ax1.set_title(f"Finger Joint Angles (Frame: {self.frame_counter})")
        
        # Refresh plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    # Initialize hand tracker
    tracker = HandTracker()
    
    # Finger configuration
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_joints = {
        "Thumb": {"A": 2, "B": 3, "C": 4},
        "Index": {"A": 5, "B": 6, "C": 8},
        "Middle": {"A": 9, "B": 10, "C": 12},
        "Ring": {"A": 13, "B": 14, "C": 16},
        "Pinky": {"A": 17, "B": 18, "C": 20}
    }
    
    # Initialize components
    try:
        live_plot = EnhancedLivePlot(fingers)
        anomaly_detector = AnomalyDetector(fingers, window_size=25, threshold=2.5)
        print("Components initialized successfully!")
    except Exception as e:
        print(f"Error initializing components: {e}")
        return
    
    # FPS tracking
    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    
    print("Hand tracking with anomaly detection started. Press ESC to exit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        
        # Process frame
        results = tracker.process_frame(frame)
        
        angle_dict = {}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get landmarks
                landmarks = tracker.get_landmarks(frame, hand_landmarks)
                
                # Calculate angles for each finger
                for finger, joints in finger_joints.items():
                    try:
                        if len(landmarks) > max(joints.values()):
                            A = landmarks[joints["A"]]
                            B = landmarks[joints["B"]]
                            C = landmarks[joints["C"]]
                            angle = calculate_angle(A, B, C)
                            angle_dict[finger] = angle
                        else:
                            angle_dict[finger] = 0
                    except (IndexError, TypeError, KeyError):
                        angle_dict[finger] = 0
        
        # Fill missing values
        for finger in fingers:
            if finger not in angle_dict:
                angle_dict[finger] = 0
        
        # Detect anomalies
        try:
            anomalies, smoothed_angles = anomaly_detector.update(angle_dict)
            
            # Update plot
            live_plot.update(angle_dict, smoothed_angles, anomalies)
        except Exception as e:
            print(f"Error in processing: {e}")
            continue
        
        # Display info on frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display anomalies
        anomaly_fingers = [finger for finger, is_anomaly in anomalies.items() if is_anomaly]
        
        if anomaly_fingers:
            cv2.putText(frame, "ANOMALY: " + ", ".join(anomaly_fingers), 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: NORMAL", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Finger Joint Anomaly Detection", frame)
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close('all')
    print("Application closed.")

if __name__ == "__main__":
    main()

