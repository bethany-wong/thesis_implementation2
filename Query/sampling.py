import cv2
import time
from Segmentation import handTrackingModuleMediaPipe as htm
import numpy as np
import json
from database import Database

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

def label_image(input_img):
    input_img = cv2.bilateralFilter(input_img, d=9, sigmaColor=75, sigmaSpace=75)
    yuv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YUV)
    labeled_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)

    # Iterate over each pixel in the YUV image
    for i in range(yuv_img.shape[0]):
        for j in range(yuv_img.shape[1]):
            pixel = yuv_img[i, j]
            min_distance = float('inf')
            label = 0
            for color_num, (lower, upper) in ranges.items():
                avg_yuv = averages[color_num]
                # Check UV ranges and Y proximity to the average Y
                if (lower[0] <= pixel[1] <= upper[0] and
                        lower[1] <= pixel[2] <= upper[1] and
                        abs(avg_yuv[0] - pixel[0]) < 30):  # Y proximity threshold
                    distance = np.linalg.norm(np.array(avg_yuv) - np.array(pixel))
                    if distance < min_distance:
                        min_distance = distance
                        label = color_num
            labeled_img[i, j] = label
    # Create an output image initialized to ones (white)
    output_img = np.ones_like(input_img)
    output_img[:, :, 0] = 255  # Set Y channel to maximum brightness
    output_img[:, :, 1:3] = 128  # Set U and V channels to neutral values

    # Fill the labeled regions with their average YUV values
    for label_num, avg_yuv in averages.items():
        mask = (labeled_img == label_num)
        output_img[mask] = avg_yuv

    # Convert the output image back to BGR for display
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_YUV2BGR)
    return output_img_bgr, labeled_img

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

def compute_centroids(labeled_img, max_blobs_per_label=2, min_pixel_count=10):
    centroids = {}
    labels = np.unique(labeled_img)
    for label in labels:
        if label == 0:
            continue
        mask = (labeled_img == label).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(mask)

        blob_areas = []
        for i in range(1, num_labels):
            y_coords, x_coords = np.where(labels_im == i)
            blob_areas.append((i, len(y_coords)))

        # Sort blobs by area in descending order and keep only the largest blobs
        blob_areas.sort(key=lambda x: x[1], reverse=True)
        blob_areas = blob_areas[:max_blobs_per_label]

        label_centroids = []
        for blob_label, area in blob_areas:
            if area < min_pixel_count:
                continue
            y_coords, x_coords = np.where(labels_im == blob_label)
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            label_centroids.append((centroid_x, centroid_y))

        centroids[label] = label_centroids
    return centroids

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
    img_skin = transform_glove_to_skin(img, True) # cover glove with skin color
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
    labeled_roi, labels_matrix = label_image(roi_resized)
    img[img.shape[0] - 40:img.shape[0], 0:40] = labeled_roi

    # compute centroids using label matrix
    centroids_in_tiny_image = compute_centroids(labels_matrix) # dictionary of labels and respective centroids

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
