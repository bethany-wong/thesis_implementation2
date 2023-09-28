import cv2
import time
from Segmentation import handTrackingModuleMediaPipe as htm
import numpy as np


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
        hsv[mask_total > 0, 2] = cv2.normalize(original_value[mask_total > 0], None, 94, 191, cv2.NORM_MINMAX).flatten()

    skin_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    skin_img = cv2.GaussianBlur(skin_img, (5, 5), 0)
    return skin_img

wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0 # previous time

detector = htm.handDetector(detectionCon=0.75)

roi_box = None # for mean shift
roi_hist = None


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # mirror the image
    img_skin = transform_glove_to_skin(img, True)
    img_skin = detector.findHands(img_skin)
    lmList = detector.findPosition(img_skin, draw=False)

    if lmList:
        x_coords = [lm[1] for lm in lmList]
        y_coords = [lm[2] for lm in lmList]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        side_length = max(x_max - x_min, y_max - y_min) # Compute the side length of the square as the maximum of width and height
        padding = int(side_length * 0.3)  # 30% of side length as padding
        x_min = max(0, x_min - padding) # adjust bounding box
        x_max = min(wCam, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(hCam, y_max + padding)

        roi = img[y_min:y_max, x_min:x_max]
        # Update the ROI box for Mean Shift
        roi_box = (x_min, y_min, x_max - x_min, y_max - y_min)
        roi = img[y_min:y_max, x_min:x_max]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    elif roi_hist is not None:
        # Use Mean Shift to estimate the new position
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, roi_box = cv2.meanShift(dst, tuple(map(int, roi_box)),
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        x_min, y_min, w, h = map(int, roi_box)
        x_max, y_max = x_min + w, y_min + h
        roi = img[y_min:y_max, x_min:x_max]
    else:
        roi = np.zeros((40, 40, 3), np.uint8)


    h, w, _ = roi.shape # Resize the ROI to fit within a bounding box of approximately four times 40x40 pixels
    aspect_ratio = w / h
    if w > h:
        new_w = 160
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = 160
        new_w = int(new_h * aspect_ratio)
    roi_resized = cv2.resize(roi, (new_w, new_h))

    # Overlay the ROI at the top left corner of the main video feed
    img[0:new_h, 0:new_w] = roi_resized

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)