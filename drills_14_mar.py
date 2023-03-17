"""
Tasks:

1) Read the image (opencv)
2) Create a rotation
3) Create a projective transformation
"""

import numpy as np
import pandas as pd
import cv2
import math

def main():
    problem_one()
    problem_two()
    problem_three()

def problem_one():
    img = cv2.imread('palazzo.jpg')
    cv2.imshow('Input', img)
    cv2.waitKey()

def problem_two():
    # Missing interpolation
    img = cv2.imread('palazzo.jpg')
    rot_matrix = np.array([[math.cos(np.pi/6), -math.sin(np.pi/6)], [math.sin(np.pi/6), math.cos(np.pi/6)]])
    canvas = np.zeros([2*img.shape[0], 2*img.shape[1], 3])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pos_vec = np.array([i, j])
            pos_vec = np.matmul(rot_matrix, pos_vec)
            pos_vec = np.floor(pos_vec).astype(int)

            # Has to be corrected
            x_padding = round(canvas.shape[0]/2 - img.shape[0]*math.cos(np.pi/3))
            y_padding = round(canvas.shape[1]/2 - img.shape[1]*math.sin(np.pi/6))

            canvas[pos_vec[0] + x_padding, pos_vec[1] + y_padding] = img[i, j]
    
    cv2.imwrite("rotated_palazzo.jpg", canvas)

def problem_three():
    img = cv2.imread('palazzo.jpg')
    rows, cols = img.shape[:2]

    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]])

    src_points = np.float32([[0,0], [0,rows-1], [cols/2,0], [cols/2,rows-1]])
    dst_points = np.float32([[0,100], [0,rows-101], [cols/2,0], [cols/2,rows-1]])

    proj_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img, proj_matrix, (cols,rows))
    cv2.imwrite('perspective_palazzo.jpg', img_output)

if __name__ == "__main__":
    main()