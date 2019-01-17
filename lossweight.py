import numpy as np 
import os
import cv2
import random
import pickle

dir_img ="ip_data/"
dir_seg ="labels/"

n_classes=10
input_height , input_width = 224 , 768
output_height , output_width = 224 , 768

images = os.listdir(dir_img)
images.sort()
segmentations  = os.listdir(dir_seg)
segmentations.sort()


NC=np.zeros((9,1))
PC=np.zeros((9,1))
FC=np.zeros((9,1))
for c in range(9):
	nc=0
	pc=0
	print(c)
	for i in range(len(images)):
		img_name=images[i]
		seg_name=img_name+'_'+str(c)+'.jpeg'
		seg=cv2.imread(dir_seg+seg_name,0)
		ret,thresh1 = cv2.threshold(seg,127,255,cv2.THRESH_BINARY)
		aa=np.count_nonzero(thresh1==255)
		if(aa>0):
			nc+=1
		pc+=aa

	NC[c]=nc
	PC[c]=pc
	if(nc==0):
		continue
	FC[c]=pc/nc


print(NC)
print(PC)
M=np.median(FC)
ALPHA=np.zeros((9,1))
for i in range(len(FC)):
	if(FC[i]==0):
		continue
	ALPHA[i]=M/FC[i]

print(ALPHA)
with open('classweights.pickle','wb') as f:
	pickle.dump(ALPHA,f)