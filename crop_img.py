import os

import numpy as np
import cv2
import pandas as pd


reshape_size = (300,300)

# x / width * 300
def normalize(tf_pos, t_pos, st_pos, roi):
    tf_pos = tf_pos/ tf_pos[2]

    tf_pos[0] = tf_pos[0] + t_pos[i,0] - st_pos[i,0]
    tf_pos[1] = tf_pos[1] + t_pos[i,1] - st_pos[i,1]

    tf_pos = tf_pos / roi.shape[0] * reshape_size[0]

    return tf_pos

data = pd.read_csv('Datasets/labels.csv')
bb_st = data[['bb_st_px', 'bb_st_py']][:1000].to_numpy()
bb_ed = data[['bb_ed_px', 'bb_ed_py']][:1000].to_numpy()

pair1 = data[['m11','m12']].to_numpy()
pair2 = data[['m21','m22']].to_numpy()
pair3 = data[['m31','m32']].to_numpy()
pair4 = data[['m13','m23']].to_numpy()

pair1 = pair1.astype(np.float32)
pair2 = pair2.astype(np.float32)
pair3 = pair3.astype(np.float32)
pair4 = pair4.astype(np.float32)


# ct_px,ct_py,st_px,st_py,ed_px,ed_py,pt_px,pt_py
center_pos = data[['ct_px','ct_py']].to_numpy()
start_pos = data[['st_px','st_py']].to_numpy()
end_pos = data[['ed_px','ed_py']].to_numpy()
pointer_pos = data[['pt_px','pt_py']].to_numpy()

center_pos = center_pos.astype(np.float32)
start_pos = start_pos.astype(np.float32)
end_pos = end_pos.astype(np.float32)
pointer_pos = pointer_pos.astype(np.float32)

#다시 라벨링할 파일 경로
f = open('relabeled_pos.txt','w')

#scale에 대한 정보가 증발해서 그냥 crop 할때 라벨링을 다시해주는게 좋을듯
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
    
    #roi 크기 결정됨

    #translataion을 행렬안에 넣었더니 값의 변화가 심해서 별도의 연산 수행
    #최초 행렬곱해질때 영향이 있을 것으로 보임
    h_matrix = np.array([[pair1[i,0],pair1[i,1], 0.],
                        [pair2[i,0],pair2[i,1], 0.],
                        [pair3[i,0],pair3[i,1], 1.]], dtype=np.float32)
    ct_pos = np.transpose(np.array([center_pos[i,0], center_pos[i,1], 1.0], dtype=np.float32))
    st_pos = np.transpose(np.array([start_pos[i,0], start_pos[i,1], 1.0], dtype=np.float32))
    ed_pos = np.transpose(np.array([end_pos[i,0], end_pos[i,1], 1.0], dtype=np.float32))
    pt_pos = np.transpose(np.array([pointer_pos[i,0], pointer_pos[i,1], 1.0], dtype=np.float32))

    ct_pos = normalize(h_matrix @ ct_pos, pair4, bb_st, roi)
    st_pos = normalize(h_matrix @ st_pos, pair4, bb_st, roi)
    ed_pos = normalize(h_matrix @ ed_pos, pair4, bb_st, roi)
    pt_pos = normalize(h_matrix @ pt_pos, pair4, bb_st, roi)

    input_txt = str(ct_pos[0]) + ',' + str(ct_pos[1]) + ',' + str(st_pos[0]) + ',' + str(st_pos[1]) + ',' + str(ed_pos[0]) + ',' + str(ed_pos[1]) + ',' + str(pt_pos[0]) + ',' + str(pt_pos[1]) + '\n'
    f.write(input_txt)

    # roi  = cv2.resize(roi,reshape_size)
    # cv2.imwrite('Datasets/crop_data/' + str(i) + '.png', roi)

f.close()
