import os
import random
#import pandas as pd
import numpy as np
import cv2

def makeHMatrix():
    #m11, m22 : 0.8 ~ 1.2
    #m12, m21 : -0.009 ~ 0.009
    #m31, m32 : -0.00075 ~ 0.00075
    m11 = random.uniform(0.8, 1.2)
    m22 = random.uniform(0.8, 1.2)
    m12 = random.uniform(-0.009, 0.009)
    m21 = random.uniform(-0.009, 0.009)
    m31 = random.uniform(-0.00075, 0.00075)
    m32 = random.uniform(-0.00075, 0.00075)

    H = np.array([[m11, m12,0.],
                  [m21, m22, 0.],
                  [m31, m32, 1.]], dtype=np.float32)

    return H

def imgMasking(img_fg, img_bg):
    #Homogenous matrix Load
    h_matrix = makeHMatrix()
    img_fg = cv2.warpPerspective(img_fg, h_matrix, (800,800))

    #Make alpha channel mask & invers
    _, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    #set ROI from background img 
    img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = img_bg[250:250+h, 250:250+w]

    #And operation
    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    #Img 합성
    added = masked_fg + masked_bg
    img_bg[250:250+h, 250:250+w] = added
    
    return img_bg, h_matrix

def saveParameter(img_class, img_size, h_matrix, scale_angle, pointer_angle):
    #class, img_size, center_pos, transformed_pos, scale_angle, pointer_angle 
    pass

def main():
    img_size = (300,300)
    img_fg = cv2.imread('Datasets/gauge/gauge_0_1-removebg-preview.png', cv2.IMREAD_UNCHANGED)
    img_fg = cv2.resize(img_fg,img_size)
    img_bg = cv2.imread('Datasets/factory/factory_3.jpg', cv2.IMREAD_COLOR)

    result, h_matrix = imgMasking(img_fg, img_bg)
    result = result[:1000, :1000]
    result = cv2.resize(result, (800,800))
    cv2.imshow('result', result)
    saveParameter(0, img_size, h_matrix, 0, 0)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()