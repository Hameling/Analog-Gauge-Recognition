import os
import random
#import pandas as pd
import numpy as np
import cv2

counter = 0
g_queue = []

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

def gaugeAngle():
    #시작 각도, 종료 각도, 포인터 각도
    gauge_angle = [[0.0, 268.0, 0.0],       #0
                    [0.0, 240.0, 0.0],      #1
                    [0.0, 268.0, 131.0],    #2
                    [0.0, 269.0, 13.0],     #3
                    [0.0, 267.0, 0.0],      #4
                    [0.0, 268.5, 130.7],    #5
                    [0.0, 269.0, 54.0],     #6
                    [0.0, 268.5, 268.5],    #7
                    [0.0, 270.0, 134.0],    #8
                    [0.0, 270.0, 107.0],]   #9

def imgMasking(img_fg, img_bg):
    #Homogenous matrix Load
    h_matrix = makeHMatrix()
    img_fg = cv2.warpPerspective(img_fg, h_matrix, (800,800))

    # 움직임이 허용 가능한 범위
    aval_height, aval_width = 250,250

    #Make alpha channel mask & invers
    _, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    #set ROI from background img 
    img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = img_bg[aval_height:aval_height+h, aval_width:aval_width+w]

    #And operation
    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    #Img 합성
    added = masked_fg + masked_bg
    img_bg[aval_height:aval_height+h, aval_width:aval_width+w] = added
    
    return img_bg, h_matrix

def mouseCallback(event, x, y, flgs, param):
    global counter, g_queue

    if event == cv2.EVENT_FLAG_LBUTTON:
        if counter == 0: #첫번째 클릭(중심 위치)
            print('Select Center Point')
            print('({},{})'.format(x,y))
            g_queue.append((x,y))
            counter += 1

        elif counter == 1: #두번째 클릭(시작 지점)
            print('Select Start Point')
            print('({},{})'.format(x,y))
            g_queue.append((x,y))
            counter += 1

        elif counter == 2: #세번째 클릭(종료 지점)
            print('Select End Point')
            print('({},{})'.format(x,y))
            g_queue.append((x,y))
            counter += 1

        elif counter == 3: #네번째 클릭(포인터 위치)
            print("Select Pointer's Position ")
            print('({},{})'.format(x,y))
            g_queue.append((x,y))
            counter += 1

def saveParameter(img_class, img_size, h_matrix, scale_angle, pointer_angle):
    #class, img_size, center_pos, transformed_pos, scale_angle, pointer_angle 
    pass

def main():
    global g_queue
    img_fg = cv2.imread('Datasets/gauge/gauge_0_1-removebg-preview.png', cv2.IMREAD_UNCHANGED)
    img_bg = cv2.imread('Datasets/factory/factory_3.jpg', cv2.IMREAD_COLOR)

    cv2.imshow('select angle', img_fg)
    cv2.setMouseCallback('select angle', mouseCallback, img_fg)
    cv2.waitKey(0)
    cv2.destroyWindow('select angle')

    print(g_queue)
    #Gauge 이미지가 배경에 비해 너무 큰경우를 보정하기 위해 이미지 크기 수정'
    #해당 이미지의 1/2 1/2 위치에 Gauge의 중심이 위치할 것으로 판단
    img_size = (500,500)
    img_fg = cv2.resize(img_fg,img_size)
    

    result, h_matrix = imgMasking(img_fg, img_bg)
    
    #해당 코드를 여기서 하는게 아니라 bg를 넘겨줄때나 합성할 때 임의의 위치 지정
    # 0 ~ 배경 크기 - 500
    result = result[:1000, :1000]

    #이미지가 화면에서 너무 크게 보일 경우 resize
    #result = cv2.resize(result, (800,800))
    cv2.imshow('result', result)
    saveParameter(0, img_size, h_matrix, 0, 0)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()