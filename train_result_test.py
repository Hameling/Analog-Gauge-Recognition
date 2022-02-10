import os
from re import L
import numpy as np
import cv2


f = open('train_result.txt','r')

txt = f.readlines()

# h_val = list(map(float, txt[1][:-1].split(',')))

# print(h_val)
# h_matrix = np.array([[h_val[0], h_val[1], 0.],
#                     [h_val[2], h_val[3], 0.],
#                     [h_val[4], h_val[5], 1.]], dtype=np.float32)

# print(h_matrix)
# img = cv2.imread('Datasets/crop_data/' + str(101) + '.png')

# cv2.imshow('test',img)
# cv2.waitKey(0)

# result_img = cv2.warpPerspective(img, np.linalg.inv(h_matrix), (500,500))

# cv2.imshow('test',result_img)
# cv2.waitKey(0)


for i in range(50):
    h_val = list(map(float, txt[i][:-1].split(',')))
    h_matrix = np.array([[h_val[0], h_val[1], 0.],
                    [h_val[2], h_val[3], 0.],
                    [h_val[4], h_val[5], 1.]], dtype=np.float32)

    img = cv2.imread('Datasets/crop_data/' + str(i) + '.png')
    result_img = cv2.warpPerspective(img, np.linalg.inv(h_matrix), (500,500))
    cv2.imwrite('result_img/' + str(i) + '.png', result_img)