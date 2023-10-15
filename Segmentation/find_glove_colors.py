# Program for sampling colors on glove

# results
averages_names = {'dark_red': (114, 195),
            'magenta': (130, 170),
            'dark_orange': (105, 205),
            'light_orange': (100, 199),
            'yellow': (96, 179),
            'dark_green': (135, 96),
            'light_green': (109, 132),
            'dark_blue': (148, 118),
            'light_blue': (172, 57),
            'purple': (158, 125)}

ranges = {'dark_red': ([111, 177], [121, 204]),
          'magenta': ([126, 150], [135, 183]),
          'dark_orange': ([100, 187], [114, 226]),
          'light_orange': ([95, 184], [105, 211]),
          'yellow': ([91, 170], [107, 197]),
          'dark_green': ([129, 82], [140, 123]),
          'light_green': ([103, 127], [121, 140]),
          'dark_blue': ([129, 110], [157, 137]),
          'light_blue': ([163, 45], [179, 79]),
          'purple': ([133, 117], [166, 132])}

averages = {
    "1": (114, 195),
    "2": (130, 170),
    "3": (105, 205),
    "4": (100, 199),
    "5": (96, 179),
    "6": (135, 96),
    "7": (109, 132),
    "8": (148, 118),
    "9": (172, 57),
    "10": (158, 125)
}

averages_y =  {1: 100,
               2: 86,
               3: 129,
               4: 154,
               5: 154,
               6: 64,
               7: 138,
               8: 63,
               9: 94,
               10: 69}


import cv2
import numpy as np


def get_uv_values(event, x, y, flags, param):
    global u_values, v_values
    if event == cv2.EVENT_LBUTTONDOWN:
        yuv_value = yuv_img[y, x]
        u_values.append(yuv_value[1])
        v_values.append(yuv_value[2])
        print(f"Y: {yuv_value[0]}, U: {yuv_value[1]}, V: {yuv_value[2]}")

def get_y_value(event, x, y, flags, param):
    global y_values
    if event == cv2.EVENT_LBUTTONDOWN:
        yuv_value = yuv_img[y, x]
        y_values.append(yuv_value[0])
        print(f"Y: {yuv_value[0]}, U: {yuv_value[1]}, V: {yuv_value[2]}")

# Load the image
img_path = '..\dataset\glove_laptop.jpg'
img = cv2.imread(img_path)

# Convert the image to YUV color space
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

color_ranges = {}
averages = {}

start_sampling = False
start_sampling_y = True

if start_sampling:
    while True:
        color_name = input("Enter the color name (or 'exit' to quit): ")
        if color_name.lower() == 'exit':
            break
        u_values = []
        v_values = []

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', get_uv_values)

        print(f"Click on the color '{color_name}'. Press any key to finish sampling")

        cv2.imshow('Image', img)
        cv2.waitKey(0)

        u_min, u_max = min(u_values), max(u_values)
        v_min, v_max = min(v_values), max(v_values)
        color_ranges[color_name] = ([u_min, v_min], [u_max, v_max])
        averages[color_name] = (sum(u_values) / len(u_values), sum(v_values) / len(v_values))

        print(f"Range for {color_name}: U({u_min}, {u_max}), V({v_min}, {v_max})")

    cv2.destroyAllWindows()
    print("Final color ranges:", color_ranges)
    print("Averages = ", averages)

if start_sampling_y: # same as upper while loop, but for Y values
    while True:
        color_name = input("Enter the color number (or 'exit' to quit): ")
        if color_name.lower() == 'exit':
            break

        y_values = []

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', get_y_value)

        print(f"Click on the color '{color_name}'. Press any key to finish sampling")

        cv2.imshow('Image', img)
        cv2.waitKey(0)

        y_min, y_max = min(y_values), max(y_values)
        average_y[color_name] = sum(y_values) / len(y_values)

    cv2.destroyAllWindows()
    print("Averages_y = ", average_y)

color_ranges = {'dark_red': ([111, 177], [121, 204]),
                'magenta': ([126, 152], [135, 181]),
                'dark_orange': ([102, 197], [112, 224]),
                'light_orange': ([96, 184], [102, 210]),
                'yellow': ([91, 170], [102, 197]),
                'dark_green': ([131, 84], [140, 113]),
                'light_green': ([103, 127], [118, 139]),
                'dark_blue': ([134, 110], [157, 131]),
                'light_blue': ([163, 46], [179, 79]),
                'purple': ([138, 121], [166, 132])}
color_ranges2 = {'dark_red': ([111, 189], [118, 203]),
                 'magenta': ([126, 150], [135, 183]),
                 'dark_orange': ([100, 187], [114, 226]),
                 'light_orange': ([95, 187], [105, 211]),
                 'yellow': ([91, 170], [107, 189]),
                 'dark_green': ([129, 82], [140, 123]),
                 'light_green': ([104, 128], [121, 140]),
                 'dark_blue': ([129, 110], [156, 137]),
                 'light_blue': ([164, 45], [179, 78]),
                 'purple': ([133, 117], [166, 128])}

averages_sampled =  {'dark_red': (114.01923076923077, 195.1153846153846),
                     'magenta': (130.4318181818182, 169.6590909090909),
                     'dark_orange': (105.24561403508773, 205.49122807017545),
                     'light_orange': (99.78846153846153, 199.34615384615384),
                     'yellow': (96.0327868852459, 179.13114754098362),
                     'dark_green': (135.36363636363637, 96.25),
                     'light_green': (109.17647058823529, 132.14117647058825),
                     'dark_blue': (147.7945205479452, 117.53424657534246),
                     'light_blue': (172.42105263157896, 56.771929824561404),
                     'purple': (157.80281690140845, 125.40845070422536)}

averages = {'dark_red': (114, 195),
            'magenta': (130, 170),
            'dark_orange': (105, 205),
            'light_orange': (100, 199),
            'yellow': (96, 179),
            'dark_green': (135, 96),
            'light_green': (109, 132),
            'dark_blue': (148, 118),
            'light_blue': (172, 57),
            'purple': (158, 125)}

for c in averages:
    averages[c] = (round(averages[c][0]), round(averages[c][1]))
#print(averages)


combined_ranges = {}
for color in color_ranges:
    u_min_combined = min(color_ranges[color][0][0], color_ranges2[color][0][0])
    v_min_combined = min(color_ranges[color][0][1], color_ranges2[color][0][1])

    u_max_combined = max(color_ranges[color][1][0], color_ranges2[color][1][0])
    v_max_combined = max(color_ranges[color][1][1], color_ranges2[color][1][1])

    combined_ranges[color] = ([u_min_combined, v_min_combined], [u_max_combined, v_max_combined])

print(combined_ranges)
