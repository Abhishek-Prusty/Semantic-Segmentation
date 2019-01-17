from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
import numpy as np 
import os
import random
import cv2
dir_img ="ip_data/"
dir_seg ="labels/"

val_img ="val_data/"
val_seg ="val_labels/"

n_classes=10
input_height , input_width = 224 , 224
output_height , output_width = 224 , 224

def data_gen(source=dir_img, target=dir_seg,bat_size=1):
    images = os.listdir(source)
    images.sort()
    segmentations  = os.listdir(target)
    segmentations.sort()

    batch_size = bat_size
    while True:
        batch = []
        batch_labels = []            
        img_name=random.choice(images)
        img=cv2.imread(dir_img+img_name,1)
        img=cv2.resize(img,(input_height,input_width))
        img=img/255.0
        temp=[]
        for i in range(10):
            seg_name=img_name+'_'+str(i)+'.jpeg'
            seg=cv2.imread(dir_seg+seg_name,0)
            seg=cv2.resize(seg,(input_height,input_width))
            ret,thresh1 = cv2.threshold(seg,127,255,cv2.THRESH_BINARY)
            thresh1=thresh1/255.0
            temp.append(thresh1)

        yield np.expand_dims(np.array(img),axis=0), [np.reshape(np.array(temp[j]),(1,input_height,input_width,1)) for j in range(10)]

gg=data_gen()
while(1):
	a,b=next(gg)
	#print(np.unique(a))
	print(np.unique(b[0]))