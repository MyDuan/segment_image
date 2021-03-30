import numpy as np

FLOAT_MAX = 3.40282e+38


class Point:
    def __init__(self, row, col, feature, label=-1):
        self.row = row
        self.col = col
        self.label = label
        self.feature = feature


class Center:
    def __init__(self, feature=[0, 0, 0]):
        self.feature = feature


class Kmeans:
    def __init__(self, img, k):
        self.img = img
        self.k = k
        self.centers = [Center() for _ in range(self.k)]
        self.last_centers = [Center() for _ in range(self.k)]
        rows, cols, channels = img.shape
        self.points = []
        for r in range(rows):
            for c in range(cols):
                self.points.append(Point(r, c, img[r][c], -1))

    def run(self, max_iteration, smallest_convergence_radius):
        self._initialize_centers()
        current_iter = 0
        while not self._is_terminate(current_iter, max_iteration, smallest_convergence_radius):
            current_iter += 1
            self._update_labels()
            self._update_centers()

    def _get_result_points(self):
        return self.points

    def _get_result_centers(self):
        return self.centers

    def _initialize_centers(self):
        random_idx = get_random_index(len(self.points) - 1, len(self.centers))
        i_center = 0
        for index in random_idx:
            self.centers[i_center].feature = self.points[index].feature
            i_center += 1

    def _update_labels(self):
        for point in self.points:
            smallestDistCenter = FLOAT_MAX
            for i_center in range(len(self.centers)):
                dist = calc_square_distance(self.centers[i_center].feature, point.feature)
                if dist < smallestDistCenter:
                    point.label = i_center
                    smallestDistCenter = dist

    def _update_centers(self):
        self.last_centers = self.centers

        for i_center in range(len(self.centers)):
            num = 0
            sums = np.zeros(3)
            for point in self.points:
                if point.label == i_center:
                    for channel in range(3):
                        sums[channel] += point.feature[channel]
                    num += 1
            for channel in range(3):
                self.centers[i_center].feature[channel] = sums[channel] / num

    def _is_terminate(self, current_iter, max_iteration, smallest_convergence_radius):
        if (current_iter >= max_iteration and check_convergence(self.centers,
                                                                self.last_centers) < smallest_convergence_radius):
            sq = 0.0
            for point in self.points:
                sq += calc_square_distance(point.feature, self.centers[point.label].feature)
            return True
        else:
            return False


def check_convergence(current_centers, last_centers):
    convergence_radius = 0
    for i_center in range(len(current_centers)):
        convergence_radius += calc_square_distance(current_centers[i_center].feature,
                                 last_centers[i_center].feature)
    return convergence_radius


def calc_square_distance(arr1, arr2):
    return (int(arr1[0]) - int(arr2[0])) ** 2 + (int(arr1[1]) - int(arr2[1])) ** 2 + (int(arr1[2]) - int(arr2[2])) ** 2


def get_random_index(max_idx, n):
    random_idx = []
    while len(random_idx) < n:
        dist = int(np.random.uniform(1, max_idx + 1) - 1)
        if dist not in random_idx:
            random_idx.append(dist)
    return random_idx
