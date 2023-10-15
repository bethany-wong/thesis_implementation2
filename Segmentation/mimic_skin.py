# incorporated into handTrackingModuleMediaPipe

import cv2
import numpy as np


def transform_glove_to_skin(image_path, adjust_intensity):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges
    colors = {
        'blue': ([100, 50, 50], [140, 255, 255]),
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
    cv2.imshow("Image", skin_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


transform_glove_to_skin('../dataset/gloves_anker.jpg', True)
transform_glove_to_skin('../dataset/gloves_anker.jpg', False)