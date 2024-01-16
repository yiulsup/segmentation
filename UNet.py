import tensorflow as tf 

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def unet(input_shape=(256, 256, 3), num_classes=2):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Contracting Path
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Expanding Path
    up5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = layers.concatenate([up5, conv3], axis=-1)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv2], axis=-1)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv1], axis=-1)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv7)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Create the U-Net model
model = unet()

data_dir = './dataset'
images_dir = os.path.join(data_dir, 'picture')
annotations_dir = os.path.join(data_dir, 'annotation')
data_gen = ImageDataGenerator(rescale=1./255)
batch_size = 100

image_data_gen = data_gen.flow_from_directory(
    images_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode=None,
    seed=42  
)

annotation_data_gen = data_gen.flow_from_directory(
    annotations_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode=None,
    seed=42
)

train_data_gen = zip(image_data_gen, annotation_data_gen)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_gen, epochs=10)

model.save("UNet.h5")
