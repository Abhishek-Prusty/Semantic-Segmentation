import numpy as np 
import cv2
import matplotlib.pyplot as plt 
np.set_printoptions(threshold=np.nan)
im1=cv2.imread("dataset1/annotations_prepped_train/0001TP_006690.png")


print(np.unique(im1))
# plt.imshow(im1)
# plt.show()