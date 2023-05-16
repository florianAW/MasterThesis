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

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

df_data = load_image_data(resize=True, resize_dim=(s.DIM_X, s.DIM_Y), normalise=True, multiclass=s.MULTICLASS)

#save_test_images(df_data)

data_train, data_test = train_test_split(df_data, test_size=0.3, random_state=42)

X_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_train['img_vec']])
X_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_test['img_vec']])

y_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_train['mask_vec']])
y_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_test['mask_vec']])


X_train = np.array([cv2.merge([img,img,img]) for img in X_train])
X_test = np.array([cv2.merge([img,img,img]) for img in X_test])


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model


input_shape = (512, 512, 3)
res_unet = build_resnet50_unet(input_shape)
res_unet.summary()

res_unet.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="sparse_categorical_accuracy")


model_file_name = s.folder_saved_models+'best_resnet_model.h5'

metric = 'val_loss'   
mc = ModelCheckpoint(model_file_name, monitor=metric, save_best_only=True, mode='min')

EPOCHS = 25
BATCH_SIZE = 24

print('Model built and compiled...')


history = res_unet.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
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