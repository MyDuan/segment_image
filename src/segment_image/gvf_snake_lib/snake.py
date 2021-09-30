from .gradient_descent_base import GradientDecentBase
from .gvf import ParamGVF, GVF
from .utils.display import display_contour, display_gvf
import numpy as np
import math
import cv2


def is_valid(max_x, max_y, radius, center):
    return radius < min(min(max_x - center[0], center[0]), min(max_y - center[1], center[1]))


def clapping(point, max_x, max_y):
    point[0] = min(max(0.0, point[0]), max_x - 1)
    point[1] = min(max(0.0, point[1]), max_y - 1)
    return point


class ParamSnake:
    def __init__(self, alpha, beta, step_size):
        self.alpha = alpha
        self.beta = beta
        self.step_size = step_size


class Contour:
    def __init__(self, max_x, max_y, radius, center, num_points):
        self.points = np.zeros((num_points, 2))
        if not is_valid(max_x, max_y, radius, center):
            print("Your Contour are out of boundary.")
            exit(-1)
        else:
            pi = 3.1415926
            theta = 2.0 * pi / num_points
            for i in range(num_points):
                self.points[i][0] = center[0] + radius * math.cos(i * theta)
                self.points[i][1] = center[1] + radius * math.sin(i * theta)

    def get_num_points(self):
        return self.points.shape[0]


class Snake(GradientDecentBase):
    def __init__(self, original_img, gvf_x, gvf_y, contour, param_snake):
        GradientDecentBase.__init__(self, param_snake.step_size)
        self.original_img_ = original_img
        num_points = contour.get_num_points()
        self.internal_force_matrix_ = np.zeros((num_points, num_points))
        self.param_snake_ = param_snake
        self.contour = contour
        self.last_contour_ = self.contour
        self.gvf_x_ = gvf_x
        self.gvf_y_ = gvf_y
        self.gvf_contour_ = np.zeros((num_points, 2))

        # cal_internal_force_matrix
        cur_points = self.contour.points
        identity = np.eye(cur_points.shape[0], cur_points.shape[0])
        identity_down = np.roll(identity, 1, axis=0)
        identity_up = np.roll(identity, -1, axis=0)
        self.deriv_1_ = identity_up - identity
        self.deriv_2_ = self.deriv_1_ - np.matmul(identity_down, self.deriv_1_)
        self.A_ = self.deriv_2_
        self.deriv_3_ = np.matmul(identity_up, self.A_) - self.A_
        self.deriv_4_ = self.deriv_3_ - np.matmul(identity_down, self.deriv_3_)
        self.B_ = self.deriv_4_
        self.internal_force_matrix_ = identity - self.param_snake_.step_size * (
                self.param_snake_.alpha * self.A_ - self.param_snake_.beta * self.B_)

    def initialize(self):
        pass

    def update(self):
        for i in range(self.contour.get_num_points()):
            clapping(self.contour.points[i], self.gvf_x_.shape[1], self.gvf_x_.shape[0])
            self.gvf_contour_[i] = self.contour.points[i]
        gvf_normalized = np.zeros((self.contour.get_num_points(), 2))
        cv2.normalize(self.gvf_contour_, gvf_normalized, -1, 1, cv2.NORM_MINMAX)
        inv = np.linalg.inv(self.internal_force_matrix_)
        new_points = np.matmul(inv, (self.contour.points + self.param_snake_.step_size * gvf_normalized))
        self.contour.points = new_points

    def compute_energy(self):
        points = self.contour.points
        delta1_square = np.power(np.matmul(self.deriv_1_, points), 2)
        delta2_square = np.power(np.matmul(self.deriv_2_, points), 2)
        abs_gvf_contour = np.abs(self.gvf_contour_)
        energy = 0
        for i in range(points.shape[0]):
            energy += delta1_square[i][0] + delta1_square[i][1] \
                      + delta2_square[i][0] + delta2_square[i][1]\
                      - abs_gvf_contour[i][0] - abs_gvf_contour[i][1]
        return energy

    def roll_back_state(self):
        self.contour = self.last_contour_

    def back_up_state(self):
        self.last_contour_ = self.contour


if __name__ == '__main__':
    img = cv2.imread("../../../sample_imgs/star.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=3, sigmaY=3)
    grad_original_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_original_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    param_gvf = ParamGVF(1e8, 1e-9)
    gvf = GVF(param_gvf, grad_original_x, grad_original_y)
    max_iteration_gvf = 20000
    gvf.run(max_iteration_gvf)
    gvf_result = gvf.get_result_gvf()
    display_gvf(gvf_result[0], gvf_result[1], 0, True)

    max_x = gvf_result[0].shape[0]
    max_y = gvf_result[1].shape[1]
    radius = min(max_x, max_y) / 3.0
    center = (max_x / 2.0, max_y / 2.0)
    num_points = 300
    contour = Contour(max_x, max_y, radius, center, num_points)
    param_snake = ParamSnake(0.1, 0.1, 0.05)
    snake_model = Snake(img, gvf_result[0], gvf_result[1], contour, param_snake)
    snake_model.run(1000)
    result_contour = snake_model.contour
    display_contour(img, result_contour, 0, True)

