# early version of find_glove_colors

import cv2
import numpy as np

def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV Value:", hsv[y, x])

img = cv2.imread('../dataset/glove_laptop.jpg')
img2 = cv2.imread('../dataset/method1_test2.png')
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

cv2.imshow("Image", img2)
cv2.setMouseCallback("Image", get_hsv_value)

cv2.waitKey(0)
cv2.destroyAllWindows()