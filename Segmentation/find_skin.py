# Program for finding HSI values to convert glove to skin color

import cv2
import numpy as np

def transform_glove_to_skin(img, hue_val, sat_val, val_val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    hsv[mask_total > 0, 0] = hue_val
    hsv[mask_total > 0, 1] = sat_val
    hsv[mask_total > 0, 2] = val_val

    skin_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    skin_img = cv2.GaussianBlur(skin_img, (5, 5), 0)
    cv2.imshow("Image", skin_img)

def update(val=0):
    hue_val = cv2.getTrackbarPos('Hue', 'Image')
    sat_val = cv2.getTrackbarPos('Saturation', 'Image')
    val_val = cv2.getTrackbarPos('Value', 'Image')
    transform_glove_to_skin(img, hue_val, sat_val, val_val)

img = cv2.imread('../dataset/glove_laptop.jpg')
cv2.namedWindow('Image')
cv2.createTrackbar('Hue', 'Image', 11, 50, update)  # Average hue from the provided values
cv2.createTrackbar('Saturation', 'Image', 95, 173, update)  # Average saturation from the provided values
cv2.createTrackbar('Value', 'Image', 150, 255, update)  # Average value from the provided values

update()
cv2.waitKey(0)
cv2.destroyAllWindows()