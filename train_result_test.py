import os
from re import L
import numpy as np
import cv2

reshape_size = (300,300)

st_idx = 0
ed_idx = 1000

def normalize(tf_pos):
    tf_pos = tf_pos/ tf_pos[2]
    return tf_pos

f = open('relabeled_pos.txt','r')
n_labels = f.readlines()
f.close()

f = open('train_result_epoch20_1000.txt','r')
txt = f.readlines()
f.close()

f = open('reaug_data.csv','w')
label = 'ct_px,ct_py,st_px,st_py,ed_px,ed_py,pt_px,pt_py\n'
f.write(label)

for i in range(ed_idx):
    n_label = list(map(float, n_labels[i][:-1].split(',')))
    ct_pos = np.array([n_label[0], n_label[1], 1.], dtype=np.float32)
    st_pos = np.array([n_label[2], n_label[3], 1.], dtype=np.float32)
    ed_pos = np.array([n_label[4], n_label[5], 1.], dtype=np.float32)
    pt_pos = np.array([n_label[6], n_label[7], 1.], dtype=np.float32)

    h_val = list(map(float, txt[i][:-1].split(',')))
    h_matrix = np.array([[h_val[0], h_val[1], 0.],
                    [h_val[2], h_val[3], 0.],
                    [h_val[4], h_val[5], 1.]], dtype=np.float32)
    h_matrix_inv = np.linalg.inv(h_matrix)

    img = cv2.imread('Datasets/crop_data/' + str(i + st_idx) + '.png')
    result_img = cv2.warpPerspective(img, h_matrix_inv, (300,300))
    # result_img = cv2.warpPerspective(img, h_matrix, (500,500))
    cv2.imwrite('final_img/' + str(i + st_idx) + '.png', result_img)

    ct_pos = normalize(h_matrix_inv @ ct_pos)
    st_pos = normalize(h_matrix_inv @ st_pos)
    ed_pos = normalize(h_matrix_inv @ ed_pos)
    pt_pos = normalize(h_matrix_inv @ pt_pos)

    input_txt = str(ct_pos[0]) + ',' + str(ct_pos[1]) + ',' + str(st_pos[0]) + ',' + str(st_pos[1]) + ',' + str(ed_pos[0]) + ',' + str(ed_pos[1]) + ',' + str(pt_pos[0]) + ',' + str(pt_pos[1]) + '\n'

    #최종 좌표를 저장하기
    f.write(input_txt)

f.close()