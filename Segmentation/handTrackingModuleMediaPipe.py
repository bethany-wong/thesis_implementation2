import cv2
import mediapipe as mp
import numpy as np

class handDetector():
    # source: https://www.youtube.com/watch?v=01sAkU_NvOY -------------------------------------------------------------
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.results = self.hands.process(imgRGB)  # Process the image using hand tracking model

        if self.results.multi_hand_landmarks:  # If there are hand landmarks
            for handLandmarks in self.results.multi_hand_landmarks:  # Iterate over each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  # List to store landmark positions

        if self.results.multi_hand_landmarks:  # If hand landmarks are detected
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)  # turn landmark coordinates from proportions to pixel values
                lmList.append([id, cx, cy])  # Append the landmark id and coordinates
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList

    # -------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform_glove_to_skin(img, adjust_intensity):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        colors = {
            'blue': ([90, 40, 40], [150, 255, 255]),
            'red1': ([0, 50, 50], [10, 255, 255]),
            'red2': ([170, 30, 50], [180, 255, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255]),
            'orange': ([10, 50, 50], [20, 255, 255]),
            'dark_green': ([40, 20, 30], [90, 255, 150]),
            'magenta': ([140, 50, 50], [170, 255, 255])
        }

        mask_total = np.zeros_like(hsv[:, :, 0])

        for color, (lower, upper) in colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            mask_total += mask

        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        hsv[mask_total > 0, 0] = 11  # Hue
        hsv[mask_total > 0, 1] = 95  # Saturation
        if adjust_intensity:
            original_value = hsv[:, :, 2]
            hsv[mask_total > 0, 2] = cv2.normalize(original_value[mask_total > 0], None, 94, 191,
                                                   cv2.NORM_MINMAX).flatten()

        skin_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        skin_img = cv2.GaussianBlur(skin_img, (5, 5), 0)
        return skin_img