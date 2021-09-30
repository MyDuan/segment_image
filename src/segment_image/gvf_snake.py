import cv2
from gvf_snake.gvf import ParamGVF, GVF
from gvf_snake.snake import Contour, ParamSnake, Snake


class GVFSnake:
    def __init__(self, img, k):
        self.img = img
        self.k = k

    def run(self):
        img = cv2.GaussianBlur(self.img, (3, 3), sigmaX=3, sigmaY=3)
        grad_original_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_original_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        param_gvf = ParamGVF(2e8, 5e-10)
        gvf = GVF(param_gvf, grad_original_x, grad_original_y)
        max_iteration_gvf = 5e2
        gvf.run(max_iteration_gvf)
        gvf_result = gvf.get_result_gvf()

        max_x = gvf_result[0].shape[0]
        max_y = gvf_result[1].shape[1]
        radius = min(max_x, max_y) / 3.0
        center = (max_x / 2.0, max_y / 2.0)
        num_points = 300
        contour = Contour(max_x, max_y, radius, center, num_points)
        param_snake = ParamSnake(0.1, 0.1, 0.05)
        snake_model = Snake(img, gvf_result[0], gvf_result[1], contour, param_snake)
        snake_model.run(10)
        result_contour = snake_model.contour
