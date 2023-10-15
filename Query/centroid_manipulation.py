# contains functions that works with centroids

import numpy as np
import cv2

class Centriod_processer:
    @staticmethod
    def compute_centroids(labeled_img, max_blobs_per_label=2, min_pixel_count=5):
        '''
        :param labeled_img: 40x40 matrix with labels from 0 to 10
        :param max_blobs_per_label: max number of patches that belong to one color
        :param min_pixel_count: smallest size (in pixel) of a color patch
        :return: dictionary of centroids in the form: label - centroids
        '''
        centroids = {}
        labels = np.unique(labeled_img)
        for label in labels:
            if label == 0:
                continue
            mask = (labeled_img == label).astype(np.uint8) # consider only pixels with current label
            num_labels, labels_im = cv2.connectedComponents(mask) # find blobs of label

            blob_areas = []
            for i in range(1, num_labels):
                y_coords, x_coords = np.where(labels_im == i) # positions of pixels in this blob
                blob_areas.append((i, len(y_coords))) # blob label and number of pixels in blob

            # Sort blobs by area in descending order and keep only the largest blobs
            blob_areas.sort(key=lambda x: x[1], reverse=True)
            blob_areas = blob_areas[:max_blobs_per_label]

            cur_label_centroids = []
            for blob_label, area in blob_areas:
                if area < min_pixel_count:
                    continue
                y_coords, x_coords = np.where(labels_im == blob_label)
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                cur_label_centroids.append((centroid_x, centroid_y))

            centroids[label] = cur_label_centroids
        return centroids

    @staticmethod
    def align_centroids(captured_centroids, neighbor_centroids):
        '''Given detected centroids of captured image and the nearest neighbor, clean the data such that the ith element
        in captured_centroids matches with the ith element in neighbor_centroids'''
        output_captured = []
        output_neighbor = []
        for label, label_centroids in captured_centroids.items():
            if len(label_centroids) > 0 and label in neighbor_centroids:
                neigh_centroids = neighbor_centroids[label]
                for cap_centroid in label_centroids:
                    if len(neigh_centroids) == 0:
                        break
                    # Find the closest centroid in neighbor centroids for the same label
                    closest_neigh_centroid = min(neigh_centroids, key=lambda x: (x[0] - cap_centroid[0]) ** 2 + (
                                x[1] - cap_centroid[1]) ** 2)
                    output_captured.append(cap_centroid)
                    output_neighbor.append(closest_neigh_centroid)
                    # Remove the matched neighbor centroid
                    neigh_centroids.remove(closest_neigh_centroid)
        return output_captured, output_neighbor

    @staticmethod
    def transform_centroids(centroids_in_tiny_image, roi_start, roi_end):
        '''Given: centroids in 40x40 tiny image, left upper corner and right bottom corner of roi;
        Return: positions of centroids in original image'''
        scaling_factor = (roi_end[0] - roi_start[0]) / 40.0 # scaling factor is the same for x and y because roi is a square
        transformed_centroids = []
        for (x, y) in centroids_in_tiny_image:
            # Scale the centroids and translate to the ROI's position in the original image
            transformed_x = int(x * scaling_factor + roi_start[0])
            transformed_y = int(y * scaling_factor + roi_start[1])
            transformed_centroids.append((transformed_x, transformed_y))
        return transformed_centroids

    @staticmethod
    def estimate_global_position(captured_centroids, neighbor_centroids, camera_matrix, actual_size=40):
        '''both centroids lists are lists of positions, no labels; actual_size: average distance between centroids of color patches in mm'''
        if len(captured_centroids) < 4: # findHomography wouldn't work
            return 0, 0, None

        captured_centroids = np.array(captured_centroids, dtype=np.float32).reshape(-1, 1, 2) # reshape for findHomography
        neighbor_centroids = np.array(neighbor_centroids, dtype=np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(neighbor_centroids, captured_centroids)
        reprojected_centroids = cv2.perspectiveTransform(neighbor_centroids.reshape(-1, 1, 2), H).reshape(-1, 2)

        centroids_array = np.array(captured_centroids) # convert back to np array
        closest_distances_captured = []
        for cent in centroids_array:
            distances = np.linalg.norm(centroids_array - cent,
                                       axis=1)  # distances from one centroid to all other centroids
            closest_distance = np.partition(distances, 1)[1]  # second smallest distance (closest one is itself)
            closest_distances_captured.append(closest_distance) # distance to closest neighboring centroid
        avg_distance_captured_pixels = np.mean(closest_distances_captured) # average of all min distances

        # Depth estimation using camera's intrinsic parameters
        focal_length_x = camera_matrix[0, 0]
        focal_length_y = camera_matrix[1, 1]
        f_avg = (focal_length_x + focal_length_y) / 2
        depth_from_camera = (f_avg * actual_size) / avg_distance_captured_pixels

        # pairwise distances between corresponding centroids
        avg_scale_factor = np.mean(np.linalg.norm(reprojected_centroids - captured_centroids, axis=1))

        return depth_from_camera, avg_scale_factor, reprojected_centroids