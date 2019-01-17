import json
from itertools import product

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2

classes = ['Hole(Virtual)','Hole(Physical)','Character Line Segment','Physical Degradation','Page Boundary','Character Component','Picture','Decorator','Library Marker']

f = open('json/sowmya.json','r')
j = f.read()
f.close()
json_data = json.loads(j)
j = json_data.get('_via_img_metadata')

kys=j.keys()
count_tr = 0
count_r = 0
for img in kys:
	all_regions = j[img]['regions']
	count_tr += 1
	if all_regions!=[]:
		count_r +=1
		path=img.split('/images/')[1]
		ab_path=path.split('-')[0]
		#print(ab_path)
		try:
			grnd=(r"bhoomi_images/images/"+ab_path)
			print(ab_path)
			grnd=grnd.replace("%20"," " )
			grnd=grnd.replace("&","" )
			S=Image.open(grnd)
			ab=ab_path.replace("/","")
			S.save('ip_data/'+ab[:-4],'jpeg')
			s1,s2=S.size
			output=np.zeros((10,s2,s1), dtype='int')
			Image_arr =[]
			for i in range(10):        
				Image_arr.append(Image.new("1", (s1, s2)))
			for r in all_regions:
				rgn=r['region_attributes']['Spatial Annotation']
				#print(rgn)
				shape=r['shape_attributes']
				ind=classes.index(rgn)
				#print('$$$$$$')
				#print(ind)
				x=shape['all_points_x']
				y=shape['all_points_y']
				vertices = tuple(zip(x,y))
				image = Image_arr[ind]  
				draw = ImageDraw.Draw(image)
				draw.polygon((vertices), outline=1, fill=1)
				img_s = np.asarray(image)
				output[ind,:,:] = img_s
				print((np.array(output)))
			for i in range(10):
				plt.imsave('llabels/'+str(i)+ab,output[i],cmap='gray')
	
		except Exception:
			pass
	
					 
	else:
		pass