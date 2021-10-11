import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = (r'D:\Datasets\Bird dataset\train')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest',)
      #validation_split=0.2)
#validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224,224),
        batch_size =10,
        class_mode='categorical',
        shuffle=True)

from tensorflow.keras.applications.resnet50 import ResNet50
model_3 = ResNet50(weights='imagenet')
for layer in model_3.layers:
  layer.trainable = False

dummy = keras.Sequential([ 
      keras.layers.Dense(900, activation = 'relu'), 
      keras.layers.Dense(300, activation= 'softmax')
])

model_VGG = keras.Sequential([
      model_3, dummy
])

print(model_VGG.summary())

model_VGG.compile ( 
      optimizer = tf.keras.optimizers.Adam(),
      loss = 'categorical_crossentropy',
      metrics=['accuracy']
)


model_VGG.fit(train_generator, epochs = 3, batch_size=10)        