"""
Tasks:

1) Create a rotation function
2) Create a projective transformation function
3) Create a function that returns the homography matrix between two images
"""

import numpy as np
import cv2
import math

def main():
    # img = cv2.imread("gol-vasco-flamengo 2023.jpg")
    # # Point 1: [340, 0] -> [340, 0]
    # # Point 2: [80, 170] -> [80, 0]
    # # Point 3: [143, 612]
    # cv2.imshow("", img[143:, 612:]/255)
    # cv2.waitKey(0)
    
    img = cv2.imread("palazzo.jpg")
    cv2.imshow("Palazzo", img/255) # imshow expects values between 0 and 1, not 0 and 255
    cv2.waitKey(0)

    rot_img = rotation(img, np.array([307, 600]), np.pi/6)
    cv2.imshow("Rotated Palazzo", rot_img/255)
    cv2.waitKey(0)

    proj_img = projection(img, np.array([307, 600]), 0, 0.001)
    cv2.imshow("Projected Palazzo", proj_img/255)
    cv2.waitKey(0)


def rotation(img, center, angle):
    trans_matrix = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    inv_rot_matrix = np.linalg.inv(rot_matrix)
    inv_trans_matrix = np.linalg.inv(trans_matrix)

    canvas = np.zeros([img.shape[0]*2, img.shape[1]*2, 3])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pos_vec = np.array([i, j, 1])
            pos_vec = np.matmul(trans_matrix, pos_vec)
            pos_vec = np.matmul(inv_rot_matrix, pos_vec)
            pos_vec = np.matmul(inv_trans_matrix, pos_vec)

            pos_vec = np.floor(pos_vec).astype(int)
            
            try:
                if pos_vec[0] < 0 or pos_vec[1] < 0:
                    pass
                else:
                    canvas[i, j] = img[pos_vec[0], pos_vec[1]]
            except:
                pass

    new_img = np.zeros([614, 1200, 3])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j] = canvas[i, j]

    return new_img

def projection(img, center, v_factor, h_factor):
    trans_matrix = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    proj_matrix = np.array([[1, 0, 0], [0, 1, 0], [v_factor, -h_factor, 1]])
    inv_proj_matrix = np.linalg.inv(proj_matrix)
    inv_trans_matrix = np.linalg.inv(trans_matrix)

    canvas = np.zeros([img.shape[0]*2, img.shape[1]*2, 3])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pos_vec = np.array([i, j, 1])
            pos_vec = np.matmul(trans_matrix, pos_vec)
            pos_vec = np.matmul(inv_proj_matrix, pos_vec)
            pos_vec = np.matmul(inv_trans_matrix, pos_vec)

            pos_vec/=pos_vec[2]

            pos_vec = np.floor(pos_vec).astype(int)
            
            try:
                if pos_vec[0] < 0 or pos_vec[1] < 0:
                    pass
                else:
                    canvas[i, j] = img[pos_vec[0], pos_vec[1]]
            except:
                pass

    new_img = np.zeros([614, 1200, 3])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j] = canvas[i, j]

    return new_img

if __name__ == "__main__":
    main()