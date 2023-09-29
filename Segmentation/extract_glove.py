import cv2
import time
from Segmentation import handTrackingModuleMediaPipe as htm
import numpy as np
from color_segmentation import Labeler

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
MIN_ROI_SIZE = 70  # Minimum side length for the ROI
HIST_DISTANCE_THRESHOLD = 0.8  # Threshold for histogram comparison

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # mirror the image
    img_skin = transform_glove_to_skin(img, True) # cover glove with skin color
    img_skin = detector.findHands(img_skin)       # use modified image to find hands using MediaPipe
    lmList = detector.findPosition(img_skin, draw=False) # get landmarks

    if lmList: # if hand was detected
        x_coords = [lm[1] for lm in lmList]
        y_coords = [lm[2] for lm in lmList]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # Compute bounding box to include all landmarks
        side_length = max(x_max - x_min, y_max - y_min)
        # adjust bounding box with 30% of side length as padding
        padding = int(side_length * 0.3)
        x_min = max(0, x_min - padding)
        x_max = min(wCam, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(hCam, y_max + padding)

        roi = img[y_min:y_max, x_min:x_max]

        roi_box = (x_min, y_min, x_max - x_min, y_max - y_min) # Update the ROI box for Mean Shift
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # binary mask for filtering pixels that belong to the hand
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180]) # calculates histogram for hue values
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX) # normalize histogram to fit in (0,255)

    elif roi_hist is not None: # if there was a bounding box in the previous frame
        # Use Mean Shift to estimate the new position
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # back projection of the histogram on to the image considering only hue (dst probability belonging to the object)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # meanshift stops either after 10 iterations or if the computed mean shift vector is smaller than 1
        ret, roi_box = cv2.meanShift(dst, tuple(map(int, roi_box)),
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        # extract new ROI
        x_min, y_min, w, h = map(int, roi_box)
        x_max, y_max = x_min + w, y_min + h
        roi = img[y_min:y_max, x_min:x_max]

        # Expand the ROI slightly after Mean Shift (10%)
        padding = int(0.1 * (x_max - x_min))
        x_min = max(0, x_min - padding)
        x_max = min(wCam, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(hCam, y_max + padding)

        # ensure minumum ROI size
        if x_max - x_min < MIN_ROI_SIZE:
            center_x = (x_max + x_min) // 2
            x_min = center_x - MIN_ROI_SIZE // 2
            x_max = center_x + MIN_ROI_SIZE // 2
        if y_max - y_min < MIN_ROI_SIZE:
            center_y = (y_max + y_min) // 2
            y_min = center_y - MIN_ROI_SIZE // 2
            y_max = center_y + MIN_ROI_SIZE // 2

        # Confidence score based on histogram comparison
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        current_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(current_hist, current_hist, 0, 255, cv2.NORM_MINMAX)
        # The Bhattacharyya distance is computed between the previously stored histogram and the current histogram
        hist_distance = cv2.compareHist(roi_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
        # if exceed a threshold, assume no hand is detected
        if hist_distance > HIST_DISTANCE_THRESHOLD:
            roi_box = None
            roi_hist = None

    else: # no hand detected, no bounding box
        roi = np.zeros((40, 40, 3), np.uint8)

    # Resize the ROI to 40x40
    roi_resized = cv2.resize(roi, (40, 40))
    img[0:40, 0:40] = roi_resized

    labeled_roi = Labeler.label_image(roi_resized)
    img[img.shape[0] - 40:img.shape[0], 0:40] = labeled_roi

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)