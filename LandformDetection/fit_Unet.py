import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import cv2
from datetime import date
from sklearn.utils import class_weight

import settings as s
from process_image_data import load_image_data, save_test_images

df_data = load_image_data(resize=True, resize_dim=(s.DIM_X, s.DIM_Y), normalise=True, multiclass=s.MULTICLASS)

#save_test_images(df_data)

data_train, data_test = train_test_split(df_data, test_size=0.3, random_state=42)

X_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_train['img_vec']])
X_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_test['img_vec']])

y_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_train['mask_vec']])
y_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_test['mask_vec']])

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
    inputs = layers.Input(shape=(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL))
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
    outputs = layers.Conv2D(s.NUMBER_CLASSES, 1, padding="same", activation = "softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    
    return unet_model

unet_model = build_unet_model()

unet_model.summary()

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="sparse_categorical_accuracy")


if s.MULTICLASS:
    model_file_name = s.folder_saved_models+'best_model_multiclass.h5'
else:
    model_file_name = s.folder_saved_models+'best_model_binary.h5'

metric = 'val_loss'   
mc = ModelCheckpoint(model_file_name, monitor=metric, save_best_only=True, mode='min')

EPOCHS = 25
BATCH_SIZE = 24

print('Model built and compiled...')


history = unet_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         verbose=1, validation_data=(X_test, y_test),
                         callbacks=[mc])

print('Model trained')

fig, (ax1, ax2) = plt.subplots(1, 2, dpi=1200, figsize=(8, 6))
fig.suptitle('U-net model with 6 objects', fontweight="bold")

ax1.set_title('Loss - Plot ')
ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Test')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Sparse Categorical Crossentropy')
ax1.set_xticks(np.arange(0, EPOCHS, step=2))

ax2.set_title('Accuracy - Plot')
ax2.plot(history.history['sparse_categorical_accuracy'])
ax2.plot(history.history['val_sparse_categorical_accuracy'])
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Sparse Categorical Accuracy')
ax2.set_xticks(np.arange(0, EPOCHS, step=2))

fig.legend(loc='upper left')
fig.tight_layout()
if s.SAVE_PLOTS:
    fig.savefig(s.folder_plots+f'U-Net_Training_History_{str(date.today())}.png', format='png')
