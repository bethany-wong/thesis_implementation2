# source: https://www.youtube.com/watch?v=01sAkU_NvOY

import cv2
import mediapipe as mp

class handDetector():
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