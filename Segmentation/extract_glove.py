import cv2
import time
from Segmentation import handTrackingModuleMediaPipe as htm
from handTrackingModuleMediaPipe import handDetector
import numpy as np
from color_segmentation import Labeler
from Query.centroid_manipulation import Centriod_processer
from Query import database


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

depth = 0.0
scale_factor = 0.0

roi_start = (0, 0)
roi_end = (0, 0)

camera_matr_L = [
    5.2508416748046875e+02, 0., 3.1661613691018283e+02,
    0., 5.1968701171875000e+02, 2.3050092757526727e+02,
    0., 0., 1.
]
camera_matrix_L = np.array(camera_matr_L).reshape(3, 3)

camera_matr_R = [
    6.1367822265625000e+02, 0., 3.2234956744316150e+02,
    0., 6.1147003173828125e+02, 2.3573998358738208e+02,
    0., 0., 1.
]
camera_matrix_R = np.array(camera_matr_R).reshape(3, 3)

db = database.Database()
hand_detected = True

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # mirror the image
    img_skin = handDetector.transform_glove_to_skin(img, True) # cover glove with skin color
    img_skin = detector.findHands(img_skin)       # use modified image to find hands using MediaPipe
    lmList = detector.findPosition(img_skin, draw=False) # get landmarks

    if lmList: # if hand was detected
        x_coords = [lm[1] for lm in lmList]
        y_coords = [lm[2] for lm in lmList]
        x_center = sum(x_coords) // len(x_coords)
        y_center = sum(y_coords) // len(y_coords)

        side_length = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))

        padding = int(side_length * 0.3)  # adjust bounding box with 30% of side length as padding
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
        roi_start = (x_min, y_min)
        roi_end = (x_max, y_max)
        hand_detected = True

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
        roi_start = (x_min, y_min)
        roi_end = (x_max, y_max)
        hand_detected = True

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
        hand_detected = False

    # Resize the ROI to 40x40
    roi_resized = cv2.resize(roi, (40, 40))
    img[0:40, 0:40] = roi_resized

    # label every pixel in roi and put labelled image in bottom left corner
    labeled_roi, labels_matrix = Labeler.label_image(roi_resized)
    img[img.shape[0] - 40:img.shape[0], 0:40] = labeled_roi

    # compute centroids using label matrix
    centroids_in_tiny_image = Centriod_processer.compute_centroids(labels_matrix)  # dictionary of labels and respective centroids

    # find neighbor and place at right bottom corner
    if hand_detected:
        neighbor_matr, neighbor_centroids = db.find_nearest_neighbor(labels_matrix, centroids_in_tiny_image)
        neighbor_img = Labeler.show_labelled_image(neighbor_matr)
    else:
        neighbor_img = np.zeros((40, 40, 3), np.uint8)
        neighbor_centroids = None
    img[img.shape[0] - 40:img.shape[0], img.shape[1] - 40:img.shape[1]] = neighbor_img

    centroids_in_tiny_image, neighbor_centroids = Centriod_processer.align_centroids(centroids_in_tiny_image, neighbor_centroids)
    centroids = Centriod_processer.transform_centroids(centroids_in_tiny_image, roi_start, roi_end)
    if neighbor_centroids != None:
        depth, scale_factor, reprojected_centroids = Centriod_processer.estimate_global_position(captured_centroids=centroids,
                                                                              neighbor_centroids=neighbor_centroids, camera_matrix=camera_matrix_R)


    # -----------------------------  Visualisation  -----------------------------------------------------------

    visualized_img = labeled_roi.copy()
    # print("centroids: ", centroids)
    for (x, y) in centroids_in_tiny_image:
        visualized_img[y, x] = [0, 255, 255]  # mark centroids with bright yellow

    visualized_img = cv2.resize(visualized_img, (visualized_img.shape[1] * 10, visualized_img.shape[0] * 10),
                                interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Centroids Visualization", visualized_img)

    cv2.putText(img, f"Depth: {round(depth, 2)} mm; Scale factor: {round(scale_factor, 2)}", (350, 100),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    for i, (x, y) in enumerate(centroids):
        cv2.circle(img, (x, y), radius=5, color=(0, 255, 255), thickness=-1)  # captured centroid in yellow
        if reprojected_centroids is not None:
            rx, ry = map(int, reprojected_centroids[i])  # Convert the coordinates to integers
            cv2.circle(img, (rx, ry), radius=5, color=(0, 255, 0), thickness=-1)  # reprojected centroid in green
            cv2.line(img, (x, y), (rx, ry), color=(0, 255, 255), thickness=1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
    cv2.imshow("Video", img)
    cv2.waitKey(1)