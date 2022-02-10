import os

import numpy as np
import cv2
import pandas as pd


data = pd.read_csv('Datasets/labels.csv')
bb_st = data[['bb_st_px', 'bb_st_py']][:1000].to_numpy()
bb_ed = data[['bb_ed_px', 'bb_ed_py']][:1000].to_numpy()


for i in range(0,1000):
    img = cv2.imread('Datasets/synthetic_data/' + str(i) + '.png')
    # roi = img[int(bb_st[i][1]):int(bb_ed[i][1]), int(bb_st[i][0]):int(bb_ed[i][0])]

    if int(bb_ed[i][1]) > int(bb_ed[i][0]):
        val = int(bb_ed[i][1]) - int(bb_st[i][1])
        roi = img[int(bb_st[i][1]):int(bb_st[i][1])+val, int(bb_st[i][0]):int(bb_st[i][0])+val]
    else:
        val = int(bb_ed[i][0]) - int(bb_st[i][0])
        roi = img[int(bb_st[i][1]):int(bb_st[i][1])+val, int(bb_st[i][0]):int(bb_st[i][0])+val]

    if roi.shape[0] != roi.shape[1]:
        if roi.shape[0] > roi.shape[1]:
            roi = roi[0:roi.shape[1], 0:roi.shape[1]]
        else : 
            roi = roi[0:roi.shape[0], 0:roi.shape[0]]
    roi  = cv2.resize(roi,(300,300))
    cv2.imwrite('Datasets/crop_data/' + str(i) + '.png', roi)
