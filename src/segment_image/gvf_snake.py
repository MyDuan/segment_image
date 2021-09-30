import cv2
from .gvf_snake_lib import ParamGVF, GVF
from .gvf_snake_lib import Contour, ParamSnake, Snake
from .gvf_snake_lib import display_contour, display_gvf


class GVFSnake:
    def __init__(self, img, max_iteration_gvf, max_iteration_snake,
                 gvf_smooth_term_weight=2e8, gvf_step_size=5e-10,
                 snake_alpha=0.1, snake_beta=0.1, snake_step_size=0.1):
        self.img = img
        self.max_iteration_gvf = max_iteration_gvf
        self.max_iteration_snake = max_iteration_snake
        self.gvf_smooth_term_weight = gvf_smooth_term_weight
        self.gvf_step_size = gvf_step_size
        self.snake_alpha = snake_alpha
        self.snake_beta = snake_beta
        self.snake_step_size = snake_step_size

    def run(self, save=False):
        img = cv2.GaussianBlur(self.img, (3, 3), sigmaX=3, sigmaY=3)
        grad_original_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_original_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        param_gvf = ParamGVF(self.gvf_smooth_term_weight, self.gvf_step_size)
        gvf = GVF(param_gvf, grad_original_x, grad_original_y)
        gvf.run(self.max_iteration_gvf)
        gvf_result = gvf.get_result_gvf()
        display_gvf(gvf_result[0], gvf_result[1], 0, save)
        max_x = gvf_result[0].shape[0]
        max_y = gvf_result[1].shape[1]
        radius = min(max_x, max_y) / 3.0
        center = (max_x / 2.0, max_y / 2.0)
        num_points = 300
        contour = Contour(max_x, max_y, radius, center, num_points)
        param_snake = ParamSnake(self.snake_alpha, self.snake_beta, self.snake_step_size)
        snake_model = Snake(img, gvf_result[0], gvf_result[1], contour, param_snake)
        snake_model.run(self.max_iteration_snake)
        result_contour = snake_model.contour
        display_contour(img, result_contour, 0, save)
        return result_contour
