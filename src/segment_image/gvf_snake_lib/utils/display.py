import cv2
import numpy as np


def draw_optical_flow(fx, fy, cflowmap, step, scaleFactor, color):
    for r in range(0, cflowmap.shape[0], step):
        for c in range(cflowmap.shape[1]):
            fxy = np.zeros(2)
            fxy[0] = fx[r][c]
            fxy[1] = fy[r][c]
            if fxy[0] != 0 or fxy[1] != 0:
                cv2.line(cflowmap, (c, r), (int(c + (fxy[0]) * scaleFactor), int(r + (fxy[1]) * scaleFactor)), color, 1, cv2.LINE_AA)
            cv2.circle(cflowmap, (c, r), 1, (255, 0, 0), 1)


def display_gvf(fx, fy, delay, save=False):
    cflowmap = np.zeros((fx.shape[0], fx.shape[1]))
    step = 8
    scaleFactor = 7
    color = (0, 255, 0)
    disp_fx = fx
    disp_fy = fy
    cv2.normalize(disp_fx, disp_fx, -1, 1, cv2.NORM_MINMAX)
    cv2.normalize(disp_fy, disp_fy, -1, 1, cv2.NORM_MINMAX)
    draw_optical_flow(disp_fx, disp_fy, cflowmap, step, scaleFactor, color)
    #cv2.imshow("img", cflowmap)
    #cv2.waitKey(delay)
    if save:
        cv2.imwrite("gvf_display.png", cflowmap)


def display_contour(img, contour, delay, save=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(contour.get_num_points() - 1):
        cv2.line(img_rgb, (int(contour.points[i][0]), int(contour.points[i][1])),
                 (int(contour.points[i+1][0]), int(contour.points[i+1][1])), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.line(img_rgb, (int(contour.points[0][0]), int(contour.points[0][1])),
             (int(contour.points[contour.get_num_points() - 1][0]), int(contour.points[contour.get_num_points() - 1][1])),
             (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("snake", img_rgb)
    cv2.waitKey(delay)
    if save:
        cv2.imwrite("snake_display.png", img_rgb)
