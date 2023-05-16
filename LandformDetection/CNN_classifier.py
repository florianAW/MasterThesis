import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import cv2
import seaborn as sns
from datetime import date
#import pydot, graphviz, pydotplus
from tensorflow.keras.utils import plot_model

import create_plots as plot
import settings as s
import process_image_data as process_data

df_data = process_data.load_image_data(resize=True, resize_dim=(s.DIM_X,s.DIM_Y), normalise=True, multiclass=True)

data_train, data_test = train_test_split(df_data, test_size=0.3, random_state=42)

X_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_train['img_vec']])
X_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL) for img in data_test['img_vec']])


y_train = data_train['label'].to_numpy()
y_test = data_test['label'].to_numpy()

y_train_onehot = np.zeros((y_train.size, y_train.max() + 1))
y_train_onehot[np.arange(y_train.size), y_train] = 1

y_test_onehot = np.zeros((y_test.size, y_test.max() + 1))
y_test_onehot[np.arange(y_test.size), y_test] = 1


def nn_setup():
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu',
                     input_shape=(128, 128, 1)))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=48, activation='relu'))
        model.add(Dense(units=20, activation='relu'))
        
        model.add(Dense(units=s.NUMBER_CLASSES, activation='softmax'))
        
        return model
    
conv_model = nn_setup()

conv_model.summary()

conv_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics="categorical_accuracy")



model_file_name = s.folder_saved_models+'best_model_CNN_classifier_C+P.h5'


metric = 'val_loss'   
mc = ModelCheckpoint(model_file_name, monitor=metric, save_best_only=True, mode='min')

EPOCHS = 20
BATCH_SIZE = 24

print('Model built and compiled...')


history = conv_model.fit(X_train, y_train_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         verbose=1, validation_data=(X_test, y_test_onehot),
                         callbacks=[mc])

print('Model trained')


fig, (ax1, ax2) = plt.subplots(1, 2, dpi=1200, figsize=(8, 6))
fig.suptitle('CNN Classifier Crater/Pits', fontweight="bold")

ax1.set_title('Loss - Plot ')
ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Test')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Categorical Crossentropy')
ax1.set_xticks(np.arange(0, EPOCHS, step=EPOCHS//10))

ax2.set_title('Accuracy - Plot')
ax2.plot(history.history['categorical_accuracy'])
ax2.plot(history.history['val_categorical_accuracy'])
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Categorical Accuracy')
ax2.set_xticks(np.arange(0, EPOCHS, step=EPOCHS//10))

fig.legend(loc='upper left')
fig.tight_layout()
if s.SAVE_PLOTS:
    fig.savefig(s.folder_plots+f'CNN_Classifier_Training_History_C+P_{str(date.today())}.png', format='png')



best_classifier = keras.models.load_model(s.folder_saved_models+'best_model_CNN_classifier_C+P.h5')

prediction = best_classifier.predict(X_test)
prediction = tf.argmax(prediction, axis=-1).numpy()

plot.Confusion_Matrix(y_truth=data_test['label'], y_pred=prediction, average_score=None, 
                      labels=['Pit', 'Crater'], save_plot=s.SAVE_PLOTS,
                      file_name='CM_CNN_C+P')