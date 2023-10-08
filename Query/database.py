import numpy as np

class Database:
    @staticmethod
    def find_nearest_neighbor(captured_image):
        centroids_palm = {1: [(22, 23)], 2: [], 3: [(23, 19)], 4: [(24, 28)], 5: [(26, 4)], 7: [(29, 1)],
                          8: [(11, 18), (28, 12)], 9: [(24, 11)], 10: [(14, 27), (20, 29)]}
        centroids_back = {1: [(27, 13), (18, 25)], 2: [(14, 23)], 3: [(21, 21), (24, 13)], 4: [(22, 16), (11, 27)],
                          5: [(25, 24)], 7: [(26, 5)], 8: [(26, 30), (16, 15)], 10: [(14, 32), (31, 17)]}
        return np.zeros((40, 40, 3), np.uint8), centroids_palm