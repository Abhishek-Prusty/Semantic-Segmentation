import numpy as np 
import cv2
import os

files=os.listdir('ip_data/')
files.sort()

for img in files:
	count=0
	for i in range(10):
		name=img+'_'+str(i)+'.jpeg'
		if(os.path.isfile('labels/'+name)):
			count+=1

	if(count !=10):
		os.remove('ip_data/'+img)
		for i in range(10):
			name=img+'_'+str(i)+'.jpeg'
			try:
				os.remove('labels/'+name)
			except Exception:
				pass