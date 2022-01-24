from glob import has_magic
import os
import random
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

def refCode():
    img_fg = cv2.imread('gauge_0-removebg-preview.png', cv2.IMREAD_UNCHANGED)
    img_fg = cv2.resize(img_fg,(300,300))
    img_bg = cv2.imread('factory_1.jpg', cv2.IMREAD_COLOR)

    h_matrix = makeHMatrix()
    img_fg = cv2.warpPerspective(img_fg, h_matrix, (800,800))
    #--② 알파채널을 이용해서 마스크와 역마스크 생성
    _, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    #--③ 전경 영상 크기로 배경 영상에서 ROI 잘라내기
    img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = img_bg[250:250+h, 250:250+w ]

    #--④ 마스크 이용해서 오려내기
    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    #--⑥ 이미지 합성
    added = masked_fg + masked_bg
    img_bg[250:250+h, 250:250+w] = added
    
    cv2.imshow('result', img_bg[:1000, :1000])

    cv2.waitKey(0)


def main():
    img = cv2.imread('gauge_0.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img,(500,500))
    bg = cv2.imread('factory_1.jpg', cv2.IMREAD_COLOR)
    
    # h_matrix = np.array([[1.,0.,0.],
    #                     [0.,1.,0.],
    #                     [0.,0.,1.]], dtype=np.float32)
    h_matrix = makeHMatrix()
    print(h_matrix)
    print(h_matrix.shape)
    print(img.shape)
    print(img.dtype)
    img = cv2.warpPerspective(img, h_matrix, (800,800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    
    h, w = img.shape[:2]
    roi = bg[250:250+h, 250:250+w ]

    #target = cv2.bitwise_and(img, thresh)

    masked_fg = cv2.bitwise_and(img, img, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    added = masked_fg + masked_bg
    bg[250:250+h, 250:250+w] = added

    cv2.imshow('target', mask)
    cv2.imshow('result', bg)

    cv2.waitKey(0)

def checkTransformResult():
    img = cv2.imread('gauge_0.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img,(500,500))
    point_val = np.array([[500],[500],[1]], dtype=np.float32)
    

    # h_matrix = np.array([[1.2,-0.009,0.],
    #                     [0.009,1.2,0.],
    #                     [0.00075,0.00075,1.]], dtype=np.float32)

    h_matrix = makeHMatrix()

    print(h_matrix)
    img = cv2.warpPerspective(img, h_matrix, (1000,1000))

    
    print(np.matmul(h_matrix, point_val))
    cv2.imshow('target', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
    #refCode()
    #checkTransformResult()