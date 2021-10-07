# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:15:44 2021

@author: Admin
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import cv2



file_test= glob.glob('./test_model/*.png')

my_model= keras.models.load_model('./weights/my_model_weights_2.h5')


keras.utils.plot_model(my_model, to_file='model.png',show_shapes=True, dpi=300)

img= cv2.imread(file_test[0],0)
# cv2.imshow('image',img)
img1=np.expand_dims(img, axis=0)
img1=np.expand_dims(img1, axis=3)

result= my_model.predict(img1/255.0)

result=result[0]

indice= np.argmax(result)

if indice==0:
    print('oui c est un hélice avec un pourcentage ', result[indice]*100,'%')
else:
    print('non c est pas un hélice avec un pourcentage', result[indice]*100,'%')
