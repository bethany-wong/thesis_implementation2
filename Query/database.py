import numpy as np
import json
import cv2
from Segmentation.color_segmentation import Labeler

class Database:
    def __init__(self, image_filename="..\dataset\labeled_roi_data.npz", centroids_filename="..\dataset\centroids_data.json"):
        self.images = self.load_db_images(image_filename)
        self.centroids = self.load_centroids(centroids_filename)

    def find_nearest_neighbor_proxy(self, captured_image, captured_centroids):
        centroids_palm = {1: [(22, 23)], 2: [], 3: [(23, 19)], 4: [(24, 28)], 5: [(26, 4)], 7: [(29, 1)],
                          8: [(11, 18), (28, 12)], 9: [(24, 11)], 10: [(14, 27), (20, 29)]}
        centroids_back = {1: [(27, 13), (18, 25)], 2: [(14, 23)], 3: [(21, 21), (24, 13)], 4: [(22, 16), (11, 27)],
                          5: [(25, 24)], 7: [(26, 5)], 8: [(26, 30), (16, 15)], 10: [(14, 32), (31, 17)]}
        return np.zeros((40, 40), np.uint8), centroids_palm

    def compute_distance(self, matrix1, matrix2):
        total_distance = 0
        for label in range(1, 11):  # For labels 1 to 10
            mask1 = (matrix1 == label)
            mask2 = (matrix2 == label)

            pixels1 = matrix1[mask1]
            pixels2 = matrix2[mask2]

            if len(pixels1) > 0 and len(pixels2) > 0:
                distances = np.abs(pixels1[:, None] - pixels2)
                min_distances = np.min(distances, axis=1)
                total_distance += np.sum(min_distances)
        return total_distance

    def find_nearest_neighbor(self, captured_image, captured_centroids):
        '''Given captured and resized image, find closest image in database and return neighbor's image and centroids'''
        print("called nn", captured_centroids)
        min_distance = float('inf')
        nearest_image = None
        nearest_centroids = None
        nearest_idx = -1

        for idx, db_image in enumerate(self.images):
            distance = self.compute_distance(captured_image, db_image)
            if distance < min_distance:
                min_distance = distance
                nearest_image = db_image
                nearest_centroids = self.centroids[idx]
                nearest_idx = idx
        print("found neighbor", idx)
        return nearest_image, nearest_centroids

    def load_db_images(self, filename):
        images = []
        with np.load(filename) as data:
            for i in range(len(data)):
                images.append(data[f'arr_{i}'])
        return images

    def load_centroids(self, filename):
        with open(filename, "r") as file:
            centroids_data = json.load(file)
        centroids_list = [{int(k): v for k, v in centroids_dict.items()} for centroids_dict in centroids_data]
        return centroids_list

    def view_database(self):
        for i in range(len(self.images)):
            output_img = Labeler.show_labelled_image(self.images[i])

            for label, positions in self.centroids[i].items():
                for (x, y) in positions:
                    output_img[y, x] = [0, 255, 255]  # mark centroids with bright yellow

            output_img_bgr = cv2.resize(output_img, (output_img.shape[1] * 10, output_img.shape[0] * 10),
                                        interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"Image {i}", output_img_bgr)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


#db = Database()
#db.view_database()
