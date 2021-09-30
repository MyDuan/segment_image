import segment_image
import cv2

if __name__ == '__main__':
    img = cv2.imread("./sample_imgs/lena.png")
    max_iteration_gvf = 1000
    max_iteration_snake = 1000
    gvf_snake = segment_image.GVFSnake(img, max_iteration_gvf, max_iteration_snake)
    gvf_snake.run(save=True)
