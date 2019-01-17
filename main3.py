import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
#import seaborn as sns
#sns.set_style("whitegrid", {'axes.grid' : False})
import random
import sys
sys.path.append("..")

import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
#from keras.utils import plot_model

from keras import optimizers
from keras.optimizers import Adam
import model_deeplab
from segmentation_models.pspnet import model as pspmodel
from keras_drop_block import DropBlock2D
from sklearn.utils import shuffle
import pandas as pd 
from keras.utils import np_utils
from imblearn.keras import BalancedBatchGenerator
warnings.filterwarnings("ignore")

dir_img ="bhoomidata/"
dir_seg ="bhoomilabels/"

val_img ="val_data/"
val_seg ="val_labels/"

n_classes=9
input_height , input_width = 224 , 768
output_height , output_width = 224 , 768

VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def FCN8( nClasses ,  input_height=224, input_width=224):
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)
    
    vgg  = Model(  img_input , pool5  )
    vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
    
    n = 4096
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
    
    
    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    # o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    cl1= Conv2DTranspose( 9 , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o1 = (Activation('sigmoid'))(cl1)

    cl2= Conv2DTranspose( 9 , kernel_size=(2,2) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(o1)
    o2 = (Activation('sigmoid'))(cl2)
    
    
    model = Model(img_input, o2)
    return model


model = FCN8(nClasses     = n_classes,  
             input_height = input_height, 
             input_width  = input_width)

# model=pspmodel.PSPNet()
print(model.summary())
#plot_model( model, show_shapes=True , to_file='model_densenet.png')
with open("classweights.pickle", "rb") as f:
    classweights= pickle.load(f)





def getImageArr( path , width , height ):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    return img

def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    ##seg_labels = np.reshap e(seg_labels,( width*height,nClasses  ))
    return seg_labels

def data_gen(source=dir_img, target=dir_seg,bat_size=1):
    images = os.listdir(source)
    images.sort()
    segmentations  = os.listdir(target)
    segmentations.sort()

    batch_size = bat_size
    while True:
        try:
            batch = []
            batch_labels = []            
            img_name=random.choice(images)
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
            temp=np.array(temp)
            temp=np.reshape(temp,(-1,input_height,input_width,9))
            #print(temp.shape)
            yield np.expand_dims(np.array(img),axis=0), np.reshape(np.array(temp),(-1,input_height,input_width,n_classes))
        except Exception as e:
            print(e)


model.compile(loss=['binary_crossentropy'],
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# loss_weights=list(classweights)
img_gen=data_gen()
val_gen=data_gen(val_img,val_seg,1)



model.fit_generator(generator=img_gen,
                    validation_data=None,
                    epochs=100,
                    steps_per_epoch=246
                    
                    )

# hist1 = model.fit(X_train,y_train,
#                   validation_data=(X_test,y_test),
#                   batch_size=32,epochs=200,verbose=1)

name='modelFCN8_bhoomi_multi-{}'.format(str(datetime.now()))
model.save(name+'.h5')

