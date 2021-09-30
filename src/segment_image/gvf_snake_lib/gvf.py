from .gradient_descent_base import GradientDecentBase
import numpy as np
import cv2


class ParamGVF:
    def __init__(self, smooth_term_weight, init_step_size):
        self.smooth_term_weight = smooth_term_weight
        self.init_step_size = init_step_size


class GVF(GradientDecentBase):
    def __init__(self, param_gvf, grad_original_x, grad_original_y):
        GradientDecentBase.__init__(self, param_gvf.init_step_size)
        self.param_gvf = param_gvf
        self.data_term_weight_ = np.zeros(grad_original_x.shape)
        self.laplacian_gvf_x_ = np.zeros(grad_original_x.shape)
        self.laplacian_gvf_y_ = np.zeros(grad_original_y.shape)

        # initial gvf external energy
        square_grad_original_x = np.power(grad_original_x, 2)
        square_grad_original_y = np.power(grad_original_y, 2)
        mag_original = np.sqrt(square_grad_original_x + square_grad_original_y)
        mag_original = cv2.GaussianBlur(mag_original, (3, 3), sigmaX=3, sigmaY=3)
        self.gvf_initial_x_ = cv2.Sobel(mag_original, cv2.CV_64F, 1, 0, ksize=3)
        self.gvf_initial_y_ = cv2.Sobel(mag_original, cv2.CV_64F, 0, 1, ksize=3)

        self.gvf_x_ = self.gvf_initial_x_
        self.gvf_y_ = self.gvf_initial_y_

        # compute the date term weight
        square_gvf_initial_x = np.power(self.gvf_initial_x_, 2)
        square_gvf_initial_y = np.power(self.gvf_initial_y_, 2)
        self.data_term_weight_ = square_gvf_initial_x + square_gvf_initial_y

    def initialize(self):
        pass

    def update(self):
        self.back_up_state()
        self.laplacian_gvf_x_ = cv2.Laplacian(self.last_gvf_x_, cv2.CV_64F)
        self.laplacian_gvf_y_ = cv2.Laplacian(self.last_gvf_y_, cv2.CV_64F)
        self.gvf_x_ = self.last_gvf_x_ + self.param_gvf.init_step_size \
                      * (self.param_gvf.smooth_term_weight * self.laplacian_gvf_x_
                      - (self.last_gvf_x_ - self.gvf_initial_x_) * self.data_term_weight_)
        self.gvf_y_ = self.last_gvf_y_ + self.param_gvf.init_step_size \
                      * (self.param_gvf.smooth_term_weight * self.laplacian_gvf_y_
                      - (self.last_gvf_y_ - self.gvf_initial_y_) * self.data_term_weight_)

    def compute_energy(self):
        gvf_ux_ = cv2.Sobel(self.gvf_x_, cv2.CV_64F, 1, 0, ksize=3)
        gvf_uy_ = cv2.Sobel(self.gvf_x_, cv2.CV_64F, 0, 1, ksize=3)
        gvf_vx_ = cv2.Sobel(self.gvf_y_, cv2.CV_64F, 1, 0, ksize=3)
        gvf_vy_ = cv2.Sobel(self.gvf_y_, cv2.CV_64F, 0, 1, ksize=3)

        square_gvf_ux_ = np.power(gvf_ux_, 2)
        square_gvf_uy_ = np.power(gvf_uy_, 2)
        square_gvf_vx_ = np.power(gvf_vx_, 2)
        square_gvf_vy_ = np.power(gvf_vy_, 2)
        sum_ = self.param_gvf.smooth_term_weight * (square_gvf_ux_ + square_gvf_uy_ + square_gvf_vx_ + square_gvf_vy_)
        smooth_term_energy = 0
        for i in range(sum_.shape[0]):
            for j in range(sum_.shape[1]):
                smooth_term_energy += sum_[i][j]

        data_term_energy = 0
        for i in range(sum_.shape[0]):
            for j in range(sum_.shape[1]):
                delta_x = self.gvf_x_[i][j] - self.gvf_initial_x_[i][j]
                delta_y = self.gvf_y_[i][j] - self.gvf_initial_y_[i][j]
                delta_f = self.data_term_weight_[i][j]
                data_term_energy += delta_f * (delta_x * delta_x + delta_y * delta_y)

        return smooth_term_energy + data_term_energy

    def roll_back_state(self):
        self.gvf_x_ = self.last_gvf_x_
        self.gvf_y_ = self.last_gvf_y_

    def back_up_state(self):
        self.last_gvf_x_ = self.gvf_x_
        self.last_gvf_y_ = self.gvf_y_

    def get_result_gvf(self):
        return [self.gvf_x_, self.gvf_y_]


if __name__ == '__main__':
    img = cv2.imread("../../../sample_imgs/lena.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=3, sigmaY=3)
    grad_original_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_original_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    param_gvf = ParamGVF(2e8, 5e-10)
    gvf = GVF(param_gvf, grad_original_x, grad_original_y)
    max_iteration_gvf = 5e2
    gvf.run(max_iteration_gvf)
