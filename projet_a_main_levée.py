# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:34:57 2021

@author: Admin
"""
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

train_datagen= keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            horizontal_flip=True,
                                                            vertical_flip=True,                                                            
                                                            rotation_range=60,
                                                            zoom_range=0.2,
                                                            validation_split=0.11)
valid_datagen= keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.11)

train_flow=train_datagen.flow_from_directory('./data/',
						target_size=(126,126),
						color_mode= 'grayscale', 
						batch_size=2,seed=40,subset='training', 
						class_mode='categorical')



valid_flow=train_datagen.flow_from_directory('./data/',
					target_size=(126,126),
					color_mode= 'grayscale', 
					batch_size=2, seed=40,
					subset='validation',
					class_mode='categorical')



print(train_flow.class_indices)

my_model= keras.Sequential()
my_model.add(keras.layers.Conv2D(8,(3,3), padding='same', activation= 'relu', input_shape=(126,126,1)))
my_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_model.add(keras.layers.Conv2D(16,(3,3), padding='same', activation= 'relu', input_shape=(126,126,1)))
my_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_model.add(keras.layers.Conv2D(16,(3,3), padding='same', activation= 'relu', input_shape=(126,126,1)))
my_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_model.add(keras.layers.Conv2D(32,(3,3), padding='same', activation= 'relu', input_shape=(126,126,1)))
my_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


# my_model.add(keras.layers.Conv2D(64,(3,3), padding='same', activation= 'relu'))
# my_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_model.add(keras.layers.Flatten())

my_model.add(keras.layers.Dense(32,activation='relu'))
# my_model.add(keras.layers.Dropout(0.1))


my_model.add(keras.layers.Dense(16,activation='relu'))
# # my_model.add(keras.layers.Dropout(0.6))




my_model.add(keras.layers.Dense(2,activation='softmax'))

my_model.summary()

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='./weights/my_model_weights_5.h5',
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True, save_weights_only=False,
                                                   mode='auto', periode=1)

my_model.compile(loss= keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adam(lr=0.00041), metrics=['accuracy'])

history=my_model.fit_generator(train_flow, epochs=50, validation_data=valid_flow, verbose=1,
                               callbacks = [model_checkpoint])



plt.figure()
plt.plot(range(1,50+1),history.history['loss'])
plt.plot(range(1,50+1),history.history['val_loss'])
plt.show()

plt.figure()
plt.plot(range(1,50+1),history.history['accuracy'])
plt.plot(range(1,50+1),history.history['val_accuracy'])
plt.show()




