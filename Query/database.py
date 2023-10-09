import numpy as np
import json
import cv2
from Segmentation.color_segmentation import Labeler
import copy

class Database:
    def __init__(self, use_mini_db=False, use_large_db=False): # mini:10, medium:50, large:100
        if use_mini_db:
            image_filename = "..\dataset\labeled_roi_data_mini.npz"
            centroids_filename = "..\dataset\centroids_data_mini.json"
        elif use_large_db:
            image_filename = "..\dataset\labeled_roi_data.npz"
            centroids_filename = "..\dataset\centroids_data.json"
        else:
            image_filename = "..\dataset\labeled_roi_data_medium.npz"
            centroids_filename = "..\dataset\centroids_data_medium.json"
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
            y1, x1 = np.where(matrix1 == label)
            y2, x2 = np.where(matrix2 == label)

            if len(x1) > 0 and len(x2) > 0:
                distances = np.sqrt((x1[:, None] - x2) ** 2 + (y1[:, None] - y2) ** 2)
                min_distances = np.min(distances, axis=1)
                total_distance += np.sum(min_distances)
        return total_distance

    def find_nearest_neighbor(self, captured_image, captured_centroids):
        '''Given captured and resized image, find closest image in database and return neighbor's image and centroids'''
        min_distance = float('inf')
        nearest_idx = -1

        for idx, db_image in enumerate(self.images):
            distance = self.compute_distance(captured_image, db_image)
            #print(f"distance from db image no.{idx}: {distance}")
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
        print("found neighbor", nearest_idx)
        #print("centroids in db: ", self.centroids[nearest_idx])
        return self.images[nearest_idx], copy.deepcopy(self.centroids[nearest_idx])

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
            print(self.centroids[i])
            for label, positions in self.centroids[i].items():
                for (x, y) in positions:
                    output_img[y, x] = [0, 255, 255]  # mark centroids with bright yellow

            output_img_bgr = cv2.resize(output_img, (output_img.shape[1] * 10, output_img.shape[0] * 10),
                                        interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"Image {i}", output_img_bgr)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def view_database_all(self): # generate an overview of the database, works for size 100
        all_images = []
        for i in range(1, len(self.images), 2):  # Step by 2 to append every 2 images
            output_img = Labeler.show_labelled_image(self.images[i])
            for label, positions in self.centroids[i].items():
                for (x, y) in positions:
                    output_img[y, x] = [0, 255, 255]  # mark centroids with bright yellow
            all_images.append(output_img)

        rows = []
        for i in range(0, len(all_images), 10):  # hstack 10 images at a time
            row = np.hstack(all_images[i:i + 10])
            rows.append(row)

        all_in_one = np.vstack(rows)  # vstack the rows
        cv2.imwrite('..\dataset\db_overview.png', all_in_one)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#db = Database(use_large_db=True)
#db.view_database_all()
