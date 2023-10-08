import cv2
import numpy as np
# filter example image
'''img = cv2.imread('..\dataset\example_seg1.jpg')
display_img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)  # Using INTER_NEAREST to keep the pixelated look
cv2.imshow('Original Image', display_img)
# d: Diameter of each pixel neighborhood. If d is non-positive, it's computed from sigmaSpace.
# sigmaColor: Filter sigma in the color space. A larger value means that farther colors within the pixel neighborhood will be mixed together.
# sigmaSpace: Filter sigma in the coordinate space. A larger value means that pixels farther away from the central pixel will influence each other.
img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
display_img2 = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)  # Using INTER_NEAREST to keep the pixelated look
cv2.imshow('Image with bilateralFilter', display_img2)
cv2.imwrite('..\dataset\example_seg1_filtered.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

class Labeler:
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

    @staticmethod
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
                for color_num, (lower, upper) in Labeler.ranges.items():
                    avg_yuv = Labeler.averages[color_num]
                    # Check UV ranges and Y proximity to the average Y
                    if (lower[0] <= pixel[1] <= upper[0] and
                        lower[1] <= pixel[2] <= upper[1] and
                        abs(avg_yuv[0] - pixel[0]) < 30):  # Y proximity threshold
                        distance = np.linalg.norm(np.array(avg_yuv) - np.array(pixel))
                        if distance < min_distance:
                            min_distance = distance
                            label = color_num
                labeled_img[i, j] = label
        return Labeler.show_labelled_image(labeled_img), labeled_img

    @staticmethod
    def show_labelled_image(labeled_img, shape=(40, 40, 3), dtype=np.uint8):
        output_img = np.ones(shape, dtype=dtype)
        output_img[:, :, 0] = 255  # Set Y channel to maximum brightness
        output_img[:, :, 1:3] = 128  # Set U and V channels to neutral values
        # Fill the labeled regions with their average YUV values
        for label_num, avg_yuv in Labeler.averages.items():
            mask = (labeled_img == label_num)
            output_img[mask] = avg_yuv
        # Convert the output image back to BGR for display
        output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_YUV2BGR)
        return output_img_bgr

# Example usage:
'''img_path = '..\dataset\example_seg1.jpg'
img = cv2.imread(img_path)
output = Labeler.label_image(img)
display_img2 = cv2.resize(output, (400, 400), interpolation=cv2.INTER_NEAREST)
cv2.imshow('Labeled Image', display_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()'''



# Visialisation of dimensions for each color patch
'''
patch_size = 100
y_values = [50, 100, 127, 150, 200]

# Create an empty white image
img = np.ones((patch_size * len(y_values), patch_size * len(ranges), 3), dtype=np.uint8) * 255

# Fill the image with color patches based on the average of the ranges for different Y values
for y_idx, y in enumerate(y_values):
    for idx, (lower, upper) in ranges.items():
        avg_u = (lower[0] + upper[0]) // 2
        avg_v = (lower[1] + upper[1]) // 2
        img[y_idx*patch_size:(y_idx+1)*patch_size, (idx-1)*patch_size:idx*patch_size] = [y, avg_u, avg_v]

# Convert from YUV to BGR for visualization
img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
for idx in ranges:
    cv2.putText(img_bgr, color_labels[idx], ((idx-1)*patch_size + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
cv2.imshow('Color Range Visualization with Different Y Values', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()'''