# experimental program, incorporated into centroid_manipulation

import cv2
import numpy as np

centroids_back = {1: [(27, 13), (18, 25)], 2: [(14, 23)], 3: [(21, 21), (24, 13)], 4: [(22, 16), (11, 27)], 5: [(25, 24)], 7: [(26, 5)], 8: [(26, 30), (16, 15)], 10: [(14, 32), (31, 17)]}
centroids_fist = {1: [(14, 23), (26, 12)], 2: [(8, 27), (6, 18)], 3: [(15, 15)], 4: [(24, 19)], 5: [(21, 25), (15, 9)], 7: [(26, 3)], 8: [(3, 27), (19, 1)], 10: [(11, 32), (24, 29)]}
centroids_palm = {1: [(22, 23)], 2: [], 3: [(23, 19)], 4: [(24, 28)], 5: [(26, 4)], 7: [(29, 1)], 8: [(11, 18), (28, 12)], 9: [(24, 11)], 10: [(14, 27), (20, 29)]}

def align_centroids(captured_centroids, neighbor_centroids):
    '''Given: detected centroids of captured image and the nearest neighbor, clean the data such that the ith element
    in captured_centroids matches with the ith element in neighbor_centroids for homography'''
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
    return np.array(output_captured, dtype=np.float32).reshape(-1, 1, 2), \
        np.array(output_neighbor, dtype=np.float32).reshape(-1, 1, 2)
def estimate_global_position(captured_centroids, neighbor_centroids, actual_size=40):
    '''actual_size: average distance between centroids of color patches in mm'''

    # match every centroid in both images
    captured_centroids, neighbor_centroids = align_centroids(captured_centroids, neighbor_centroids)
    H, _ = cv2.findHomography(neighbor_centroids, captured_centroids)
    reprojected_centroids = cv2.perspectiveTransform(neighbor_centroids.reshape(-1, 1, 2), H).reshape(-1, 2)

    centroids_array = np.array(captured_centroids)
    closest_distances_captured = []
    for cent in centroids_array:
        distances = np.linalg.norm(centroids_array - cent, axis=1) # distances from one centroid to all other centroids
        closest_distance = np.partition(distances, 1)[1] # second smallest distance (closest one is itself)
        closest_distances_captured.append(closest_distance)
    avg_distance_captured = np.mean(closest_distances_captured)
    depth = actual_size / avg_distance_captured

    # pairwise distances between corresponding centroids
    avg_scale_factor = np.mean(np.linalg.norm(reprojected_centroids - captured_centroids, axis=1))

    return depth, avg_scale_factor

print(estimate_global_position(captured_centroids=centroids_back, neighbor_centroids=centroids_fist))
