# Program for creating a database, put list of labelled matrices into npz file and their centroids into a json file
# Whole structure similar to extract_glove but only until labelling and finding centroids of ROI

import cv2
import time
from Segmentation import handTrackingModuleMediaPipe as htm
import numpy as np
import json
from database import Database
from Segmentation.color_segmentation import Labeler
from Segmentation.handTrackingModuleMediaPipe import handDetector
from Query.centroid_manipulation import Centriod_processer

averages = {  # in YUV
        1: (100, 114, 195),
        2: (86, 130, 170),
        3: (129, 105, 205),
        4: (154, 100, 199),
        5: (154, 96, 179),
        6: (64, 135, 96),
        7: (138, 109, 132),
        8: (63, 148, 118),
        9: (94, 172, 57),
        10: (69, 158, 125)
    }

ranges = {
    1: ([111, 177], [121, 204]),
    2: ([126, 150], [135, 183]),
    3: ([100, 187], [114, 226]),
    4: ([95, 184], [105, 211]),
    5: ([91, 170], [107, 197]),
    6: ([129, 82], [140, 123]),
    7: ([103, 127], [121, 140]),
    8: ([129, 110], [157, 137]),
    9: ([163, 45], [179, 79]),
    10: ([133, 117], [166, 132])
}

color_labels = {
    1: 'dark_red',
    2: 'magenta',
    3: 'dark_orange',
    4: 'light_orange',
    5: 'yellow',
    6: 'dark_green',
    7: 'light_green',
    8: 'dark_blue',
    9: 'light_blue',
    10: 'purple'
}

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0 # previous time

detector = htm.handDetector(detectionCon=0.75)

roi_box = None # for mean shift
roi_hist = None
MIN_ROI_SIZE = 70  # Minimum side length for the ROI
HIST_DISTANCE_THRESHOLD = 0.8  # Threshold for histogram comparison

db = Database()
labeled_roi_list = []
centroids_list = []


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # mirror the image
    img_skin = handDetector.transform_glove_to_skin(img, True) # cover glove with skin color
    img_skin = detector.findHands(img_skin)       # use modified image to find hands using MediaPipe
    lmList = detector.findPosition(img_skin, draw=False) # get landmarks

    if lmList:  # if hand was detected
        x_coords = [lm[1] for lm in lmList]
        y_coords = [lm[2] for lm in lmList]
        x_center = sum(x_coords) // len(x_coords)
        y_center = sum(y_coords) // len(y_coords)

        side_length = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))

        padding = int(side_length * 0.3) # adjust bounding box with 30% of side length as padding
        side_length += 2 * padding

        x_min = max(0, x_center - side_length // 2)
        x_max = x_min + side_length
        y_min = max(0, y_center - side_length // 2)
        y_max = y_min + side_length

        if x_max > wCam:
            x_max = wCam
            x_min = wCam - side_length
        if y_max > hCam:
            y_max = hCam
            y_min = hCam - side_length

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

        # Expand the ROI after Mean Shift (30%)
        padding = int(0.3 * (x_max - x_min))
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

    # label every pixel in roi and put labelled image in bottom left corner
    labeled_roi, labels_matrix = Labeler.label_image(roi_resized)
    img[img.shape[0] - 40:img.shape[0], 0:40] = labeled_roi

    # compute centroids using label matrix
    centroids_in_tiny_image = Centriod_processer.compute_centroids(labels_matrix) # dictionary of labels and respective centroids

    # -----------------------------  Visualisation  -----------------------------------------------------------
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    visualized_img = labeled_roi.copy()
    # print("centroids: ", centroids)
    for label, positions in centroids_in_tiny_image.items():
        for (x, y) in positions:
            visualized_img[y, x] = [0, 255, 255]  # mark centroids with bright yellow

    visualized_img = cv2.resize(visualized_img, (visualized_img.shape[1] * 10, visualized_img.shape[0] * 10),
                                interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Centroids Visualization", visualized_img)

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))

    cv2.imshow("Video", img)
    key = cv2.waitKey(1)
    if key == 32:  # ASCII for space key
        labeled_roi_list.append(labels_matrix)
        print(f"appended {labels_matrix}")
        centroids_list.append(centroids_in_tiny_image)
        print(f"appended {centroids_in_tiny_image}")
        print(len(labeled_roi_list))
        print(len(centroids_list))
    elif key == ord('q'):
        break

np.savez("..\dataset\labeled_roi_data.npz", *labeled_roi_list)
centroids_list_str_keys = [{str(k): v for k, v in centroids_dict.items()} for centroids_dict in centroids_list]
with open("..\dataset\centroids_data.json", "w") as file:
    json.dump(centroids_list_str_keys, file)
cap.release()
cv2.destroyAllWindows()
