import cv2
import numpy as np

from src.k_means import Kmeans

if __name__ == '__main__':
    img = cv2.imread("./sample_imgs/lena.png")
    k = 2
    kmeans = Kmeans(img, k)

    iteration = 10
    convergence_radius = 1e-6

    kmeans.run(iteration, convergence_radius)
    points = kmeans._get_result_points()
    centers = kmeans._get_result_centers()
    result = np.zeros(img.shape, dtype="uint8")
    for point in points:
        for channel in range(3):
            result[point.row][point.col][channel] = centers[point.label].feature[channel]
    re = cv2.hconcat([img, result])
    cv2.imshow("kmeans result", re)
    cv2.waitKey(0)



