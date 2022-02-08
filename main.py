import os
import random
#import pandas as pd
import numpy as np
import cv2

counter = 0
g_queue = []
t_queue = []

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

#gauge의 각도를 측정하여둔 함수
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

#사용자로부터 gauge에 대한 정보를 입력받는 함수
#gauge 시작/끝 좌표, pointer 좌표, 중심좌표
def basicGauageData(dataset_path='Datasets/rbg_gauge'):
    global counter, g_queue, t_queue
    for i, val in enumerate(os.listdir(dataset_path)):
        
        img_fg = cv2.imread(dataset_path + '/' + val, cv2.IMREAD_UNCHANGED)    
        img_fg = cv2.resize(img_fg, (500,500)) 

        cv2.imshow('select angle', img_fg)
        cv2.setMouseCallback('select angle', mouseCallback, img_fg)
        cv2.waitKey(0)

        counter = 0
        g_queue.append(t_queue)
        t_queue = []

    print(g_queue)
    f = open('Datasets/prior.txt', 'w')
    for i in g_queue:
        f.write(str(i[0][0]) + ',' + str(i[0][1]) + ',' + str(i[1][0]) + ',' + str(i[1][1]) + ',' + str(i[2][0]) + ',' + str(i[2][1]) + ',' + str(i[3][0]) + ',' + str(i[3][1]) + '\n')
    f.close()

def imgMasking(img_fg, img_bg):
    #Homogenous matrix Load
    h_matrix = makeHMatrix()
    img_fg = cv2.warpPerspective(img_fg, h_matrix, (700,700))

    # 움직임이 허용 가능한 범위
    minWH = 300
    aval_height = random.randint(0, minWH - 1)
    aval_width = random.randint(0, minWH - 1)

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
    
    h_matrix[0][2] = aval_width
    h_matrix[1][2] = aval_height
    print(h_matrix)

    return img_bg, h_matrix

def mouseCallback(event, x, y, flgs, param):
    global counter, t_queue
    if event == cv2.EVENT_FLAG_LBUTTON:
        if counter == 0: #첫번째 클릭(중심 위치)
            print('Select Center Point')
            print('({},{})'.format(x,y))
            t_queue.append((x,y))
            counter += 1

        elif counter == 1: #두번째 클릭(시작 지점)
            print('Select Start Point')
            print('({},{})'.format(x,y))
            t_queue.append((x,y))
            counter += 1

        elif counter == 2: #세번째 클릭(종료 지점)
            print('Select End Point')
            print('({},{})'.format(x,y))
            t_queue.append((x,y))
            counter += 1

        elif counter == 3: #네번째 클릭(포인터 위치)
            print("Select Pointer's Position ")
            print('({},{})'.format(x,y))
            t_queue.append((x,y))

            cv2.destroyWindow('select angle')

def saveParameter(fp, img_class, h_matrix, prior_info, test=False):
    #label = 'class, bb_st_px, bb_st_py, bb_ed_px, bb_end_py, m11, m12, m21, m22, m31, m32, ct_px, ct_py, st_px, st_py, ed_px, ed_py, pt_px, pt_py'
    
    #bb_pos 계산
    left_top = np.array([[0],[0],[1]], dtype=np.float32)
    right_top = np.array([[500],[0],[1]], dtype=np.float32)
    left_bt = np.array([[0],[500],[1]], dtype=np.float32)
    right_bt = np.array([[500],[500],[1]], dtype=np.float32)

    basic_pos = np.array([[0., 500. ,0., 500.],
                        [0., 0., 500., 500.],
                        [1., 1., 1., 1.]], dtype=np.float32)
    
    tf_pos = h_matrix @ basic_pos
    
    print(tf_pos)

    #homogenous constraint
    tf_pos[:,0] = tf_pos[:,0] / tf_pos[:,0][-1]
    tf_pos[:,1] = tf_pos[:,1] / tf_pos[:,1][-1]
    tf_pos[:,2] = tf_pos[:,2] / tf_pos[:,2][-1]
    tf_pos[:,3] = tf_pos[:,3] / tf_pos[:,3][-1]
    
    print(tf_pos)

    df_tf_pos = [0.,0.,1000.,1000.]
    if tf_pos[0].min() > 0.:
        df_tf_pos[0] = tf_pos[0].min()
    if tf_pos[1].min() > 0.:
        df_tf_pos[1] = tf_pos[1].min()

    if tf_pos[0].max() > 1000.:
        df_tf_pos[2] = tf_pos[0].max()
    if tf_pos[1].max() > 1000.:
        df_tf_pos[3] = tf_pos[1].max()

    if test:
        input_txt = (str(img_class) + ',' +
                str(df_tf_pos[0]) + ',' +  str(df_tf_pos[1]) + ',' + str(df_tf_pos[2]) + ',' + str(df_tf_pos[3]) + ',' + 
                str(h_matrix[0][0]) + ',' + str(h_matrix[0][1]) + ',' + str(h_matrix[1][0]) + ',' + str(h_matrix[1][1]) + ',' + str(h_matrix[2][0]) + ',' + str(h_matrix[2][1]) + ',' +
                str(prior_info[0][0]) + ',' + str(prior_info[0][1]) + ',' + str(prior_info[1][0]) + ',' + str(prior_info[1][1]) + ',' +
                str(prior_info[2][0]) + ',' + str(prior_info[2][1]) + ',' + str(prior_info[3][0]) + ',' + str(prior_info[3][1])) + '\n'
    else:
        input_txt = (str(img_class) + ',' +
            str(df_tf_pos[0]) + ',' +  str(df_tf_pos[1]) + ',' + str(df_tf_pos[2]) + ',' + str(df_tf_pos[3]) + ',' + 
            str(h_matrix[0][0]) + ',' + str(h_matrix[0][1]) + ',' + str(h_matrix[1][0]) + ',' + str(h_matrix[1][1]) + ',' + str(h_matrix[2][0]) + ',' + str(h_matrix[2][1]) + ',' + prior_info)
        

    print(input_txt)
    fp.write(input_txt)

    
def main():
    global t_queue
    img_fg = cv2.imread('Datasets/rbg_gauge/gauge_0.png', cv2.IMREAD_UNCHANGED)
    img_bg = cv2.imread('Datasets/factory/factory_0.jpg', cv2.IMREAD_COLOR)

    cv2.imshow('select angle', img_fg)
    cv2.setMouseCallback('select angle', mouseCallback, img_fg)
    cv2.waitKey(0)
    cv2.destroyWindow('select angle')
    

    print(t_queue)
    #Gauge 이미지가 배경에 비해 너무 큰경우를 보정하기 위해 이미지 크기 수정'
    #해당 이미지의 1/2 1/2 위치에 Gauge의 중심이 위치할 것으로 판단
    #위의 가정은 무시되고, 사용자가 최초에 입력해주는 것으로 함
    img_size = (500,500)
    img_fg = cv2.resize(img_fg,img_size) 
    

    result, h_matrix = imgMasking(img_fg, img_bg)
    
    #해당 코드를 여기서 하는게 아니라 bg를 넘겨줄때나 합성할 때 임의의 위치 지정
    # 0 ~ 배경 크기 - 500
    result = result[:1000, :1000]

    #이미지가 화면에서 너무 크게 보일 경우 resize
    #result = cv2.resize(result, (800,800))
    cv2.imshow('result', result)

    f = open('Datasets/labels.txt', 'w')
    saveParameter(f, 0, h_matrix, t_queue, True)
    f.close()
    cv2.waitKey(0)

    cv2.imwrite('test.png', result)

    cv2.destroyAllWindows()


def makeDataset(dataset_size=100000):
    fg_path = 'Datasets/rbg_gauge'
    bg_path = 'Datasets/factory_all'
    max_bg_size = len(os.listdir(bg_path))

    f = open('Datasets/prior.txt', 'r')
    gauge_info = f.readlines()
    f.close()

    f = open('Datasets/labels.txt', 'w')
    label = 'class, bb_st_px, bb_st_py, bb_ed_px, bb_end_py, m11, m12, m21, m22, m31, m32, ct_px, ct_py, st_px, st_py, ed_px, ed_py, pt_px, pt_py\n'
    f.write(label)

    for i in range(0,dataset_size):
        r_idx = random.randint(0, max_bg_size - 1)
        gauge_idx = random.randint(0, 10 - 1)
        fg_name =  os.listdir(fg_path)[gauge_idx]

        img_fg = cv2.imread(fg_path + '/' + fg_name, cv2.IMREAD_UNCHANGED)
        img_bg = cv2.imread(bg_path + '/' + str(r_idx) + '.png', cv2.IMREAD_COLOR)

        img_size = (500,500)
        img_fg = cv2.resize(img_fg,img_size) 

        #여기서 배경 crop하는 코드가 필요하네
        bg_st_h = random.randint(0, img_bg.shape[0] - 1000)
        bg_st_w = random.randint(0, img_bg.shape[1] - 1000)
        result, h_matrix = imgMasking(img_fg, img_bg[bg_st_h:bg_st_h+1000, bg_st_w:bg_st_w+1000])

        saveParameter(f, 0, h_matrix, gauge_info[gauge_idx])

        cv2.imwrite('Datasets/synthetic_data/' + str(i) + '.png', result)

    f.close()

if __name__ == '__main__':
    #main()
    #preprocessing for data labeling
    #basicGauageData()
    makeDataset(10)