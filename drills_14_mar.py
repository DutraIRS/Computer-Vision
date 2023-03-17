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

# img = cv2.imread('1200px-Palazzo_Farnese_Fassade.jpg')

# #rotate by 30 degrees
# rotate = np.array([[math.cos(np.pi/6), -math.sin(np.pi/6)], [math.sin(np.pi/6), math.cos(np.pi/6)]])

# print(img)

# imagem2 = np.zeros((2*img.shape[0], 2*img.shape[1], 3), np.uint8)

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         x = np.array([i, j])
#         x = np.dot(x, rotate)
#         x = np.floor(x).astype(int)
#         imagem2[x[0], x[1]+img.shape[1]] = img[i, j]

# cv2.imwrite("changed.jpg", imagem2)



# img1 = cv2.imread('1200px-Palazzo_Farnese_Fassade.jpg')
# rows, cols = img1.shape[:2]

# src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
# dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]])

# src_points = np.float32([[0,0], [0,rows-1], [cols/2,0], [cols/2,rows-1]])
# dst_points = np.float32([[0,100], [0,rows-101], [cols/2,0], [cols/2,rows-1]])

# projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
# img_output = cv2.warpPerspective(img1, projective_matrix, (cols,rows))

# cv2.imshow('Input', img1)
# cv2.imshow('Output', img_output)
# cv2.waitKey()


img = cv2.imread('1200px-Palazzo_Farnese_Fassade.jpg')

#perpective from left to right


perpective_matrix = np.array([[1, 0, 100], [0, 1, -100], [100,-100,1]])

imagem2 = np.zeros((3*img.shape[0], 3*img.shape[1], 3), np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        x = np.array([i, j, 1])
        x = np.matmul(x, perpective_matrix)
        x = np.floor(x).astype(int)
        imagem2[x[0], x[1]] = img[i, j]
    
cv2.imwrite("change.jpg", imagem2)