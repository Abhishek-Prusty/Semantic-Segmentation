import pickle
import numpy as np
import random
import cv2
import os
from keras.utils import np_utils

dir_img ="bhoomidata/"
dir_seg ="bhoomilabels/"

n_classes=10
input_height , input_width = 224 , 768
output_height , output_width = 224 , 768


def data_gen(source=dir_img, target=dir_seg,bat_size=1):
    images = os.listdir(source)
    images.sort()
    segmentations  = os.listdir(target)
    segmentations.sort()
    #print(len(images))
    data=[]
    labels=[]
    for j in range(len(images)):
        print(j)
        batch = []
        batch_labels = []            
        img_name=images[j]
        img=cv2.imread(dir_img+img_name,1)
        img=cv2.resize(img,(input_width,input_height))
        img=img/255.0
        temp=[]
        for i in range(9):
            seg_name=img_name+'_'+str(i)+'.jpeg'
            seg=cv2.imread(dir_seg+seg_name,0)
            seg=cv2.resize(seg,(input_width,input_height))
            ret,thresh1 = cv2.threshold(seg,127,255,cv2.THRESH_BINARY)
            thresh1=thresh1/255.0
            temp.append(thresh1.astype(int))
        data.append(np.expand_dims(np.array(img),axis=0))
        labels.append([np_utils.to_categorical(np.reshape(np.array(temp[k]),(1,input_height,input_width,1)),num_classes=9) for k in [0,1,2,3,4,5,6,7,8]])
    return np.array(data),np.array(labels)

X,y=data_gen()
print(X.shape)
print(y.shape)
print(y[0])

with open('data_bhoomi.pickle', 'wb') as f:
    pickle.dump(X,f)

with open('labels_bhoomi.pickle', 'wb') as f:
    pickle.dump(y,f)