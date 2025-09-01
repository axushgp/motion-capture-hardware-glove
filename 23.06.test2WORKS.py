import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, modelComplexity=1, trackCon=0.7):
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
        self.drawSpecLandmarks = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        self.drawSpecConnections = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    image, handLms, self.mpHands.HAND_CONNECTIONS,
                    self.drawSpecLandmarks, self.drawSpecConnections
                )
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmList = []
        if (self.results and self.results.multi_hand_landmarks and
                handNo < len(self.results.multi_hand_landmarks)):
            hand = self.results.multi_hand_landmarks[handNo]
            h, w, c = image.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id in [4, 8, 12, 16, 20]:
                    cv2.circle(image, (cx, cy), 8, (255, 255, 0), cv2.FILLED)
        return lmList

def calculate_angle(A, B, C):
    """Calculate the angle at point B given three points A, B, C as (x, y)."""
    if not (isinstance(A, tuple) and isinstance(B, tuple) and isinstance(C, tuple)):
        return 0.0
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    mag_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    mag_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    if mag_BA == 0 or mag_BC == 0:
        return 0.0
    cosine_angle = max(min(dot_product / (mag_BA * mag_BC), 1.0), -1.0)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

class LiveAnglePlot:
    def __init__(self, fingers, max_len=60):
        self.fingers = fingers
        self.data = {finger: deque([0]*max_len, maxlen=max_len) for finger in fingers}
        self.max_len = max_len

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.lines = {}
        for finger in fingers:
            (line,) = self.ax.plot([], [], label=finger)
            self.lines[finger] = line

        self.ax.set_ylim(0, 180)
        self.ax.set_xlim(0, max_len)
        self.ax.set_title("Live Finger Joint Angles")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (degrees)")
        self.ax.legend(loc='upper right')

    def update(self, angle_dict):
        for finger in self.fingers:
            self.data[finger].append(angle_dict.get(finger, 0))
            self.lines[finger].set_ydata(list(self.data[finger]))
            self.lines[finger].set_xdata(range(len(self.data[finger])))

        self.ax.relim()
        self.ax.autoscale_view(scaley=True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    tracker = HandTracker()
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_landmarks = {
        "Thumb": {"A": 2, "B": 3, "C": 4},
        "Index": {"A": 5, "B": 6, "C": 7},
        "Middle": {"A": 9, "B": 10, "C": 11},
        "Ring": {"A": 13, "B": 14, "C": 15},
        "Pinky": {"A": 17, "B": 18, "C": 19}
    }

    live_plot = LiveAnglePlot(fingers)

    while True:
        success, image = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        image = cv2.resize(image, (800, 600))
        image = cv2.flip(image, 1)
        image = tracker.handsFinder(image)
        lmLists = [tracker.positionFinder(image, handNo=i) for i in range(tracker.maxHands)]

        for lmList in lmLists:
            if not lmList:
                continue
            angle_dict = {}
            for finger, ids in finger_landmarks.items():
                try:
                    A = (lmList[ids["A"]][1], lmList[ids["A"]][2])
                    B = (lmList[ids["B"]][1], lmList[ids["B"]][2])
                    C = (lmList[ids["C"]][1], lmList[ids["C"]][2])
                    angle = calculate_angle(A, B, C)
                    angle_dict[finger] = angle
                except Exception as e:
                    angle_dict[finger] = 0
            live_plot.update(angle_dict)

        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()