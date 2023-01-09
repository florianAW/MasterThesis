import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

from process_image_data import load_image_data, save_test_images

DIM_X, DIM_Y, COLOR_CHANNEL = 128, 128, 1

df_data = load_image_data(resize=True, resize_dim=(DIM_X, DIM_Y), normalise=True)

#save_test_images(df_data)

X_train, X_test, y_train, y_test = train_test_split(
                                    df_data['img_vec'], df_data['mask_vec'], test_size=0.3, random_state=42)


X_test = np.array([img.numpy().reshape(DIM_X, DIM_Y, COLOR_CHANNEL) for img in X_test])
X_train = np.array([img.numpy().reshape(DIM_X, DIM_Y, COLOR_CHANNEL) for img in X_train])


y_test = np.array([img.numpy().reshape(DIM_X, DIM_Y, COLOR_CHANNEL) for img in y_test])
y_train = np.array([img.numpy().reshape(DIM_X, DIM_Y, COLOR_CHANNEL) for img in y_train])



#y_train = np.array([np.asarray(x).astype(np.float32) for x in y_train])
#y_test = np.array([np.asarray(x).astype(np.float32) for x in y_test])
#X_train = np.array([np.asarray(x).astype(np.float32) for x in X_train])
#X_test = np.array([np.asarray(x).astype(np.float32) for x in X_test])

#y_train = tf.keras.utils.to_categorical(y_train, num_classes=2, dtype='float32')
#y_test = tf.keras.utils.to_categorical(y_test, num_classes=2, dtype='float32')

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(DIM_X, DIM_Y, COLOR_CHANNEL))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(2, 1, padding="same", activation = "softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    
    return unet_model

unet_model = build_unet_model()

unet_model.summary()

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="sparse_categorical_accuracy")

mc = ModelCheckpoint('best_model_2.h5', monitor='val_accuracy', save_best_only=True, mode='max')

EPOCHS = 20
BATCH_SIZE = 24

print('Model built and compiled...')


model_history = unet_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   verbose=1, validation_data=(X_test, y_test),
                                   callbacks=[mc])

print('Model trained')




