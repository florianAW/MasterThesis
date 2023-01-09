import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
from sklearn.model_selection import train_test_split

from process_image_data import load_image_data, save_test_images

df_data = load_image_data(resize=True, resize_dim=(128,128), normalise=True)

X_train, X_test, y_train, y_test = train_test_split(
                                    df_data['img_vec'], df_data['mask_vec'], test_size=0.3, random_state=42)

### load u-net model
unet_model = keras.models.load_model('best_model.h5')

#unet_model.summary()

X_test = np.array([img.numpy().reshape(128, 128, 1) for img in X_test])
X_train = np.array([img.numpy().reshape(128, 128, 1) for img in X_train])


#y_test = np.array([img.numpy().reshape(128, 128, 1) for img in y_test])
#y_train = np.array([img.numpy().reshape(128, 128, 1) for img in y_train])


#score = unet_model.evaluate(X_test, y_test, verbose=1)

output_folder = 'data/prediced_masks/'

predicted_mask = unet_model.predict(X_test)
predicted_mask = tf.argmax(predicted_mask, axis=-1).numpy()


for i in range(0,predicted_mask.shape[0]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(tf.keras.utils.array_to_img(X_test[i]), cmap='gray')
    ax2.imshow(tf.keras.utils.array_to_img(y_test[i].numpy().reshape(128,128,1), scale=True), cmap='gray')
    ax3.imshow(tf.keras.utils.array_to_img(predicted_mask[i].reshape(128,128,1), scale=True), cmap='gray')
    
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    
    ax1.set_title('Real Image')
    ax2.set_title('Real Mask')
    ax3.set_title('Predicted Mask')
    
    plt.savefig(output_folder+f'comparison_{i}_.png', dpi=900, format='png')
    #plt.show()

'''
for i in range(predicted_mask.shape[0]):
    
    cv2.imwrite(output_folder+'real_image_.png', X_test[5].numpy()*255)
    cv2.imwrite(output_folder+'real_mask_1.png', y_test[5].numpy()*255)
    cv2.imwrite(output_folder+'prediced_mask_1.png', predicted_mask.numpy().reshape(128,128)*255)
'''

### multiclass

output_folder = 'data/predicted_masks_classes/'

colors = {
            1: [255,   0,   0],
            2: [255, 128,   0],
            3: [255, 255,   0],
            4: [0,   255,   0],
            5: [0,     0, 255]
            }


predicted_mask = unet_model.predict(X_test)
predicted_mask = tf.argmax(predicted_mask, axis=-1).numpy()

predicted_mask = [image.reshape(128,128,1) for image in predicted_mask]
predicted_mask = np.array([cv2.merge([img,img,img]) for img in predicted_mask])

for z in range(len(predicted_mask)):
    for y in range(len(predicted_mask[0])):
        for x in range(len(predicted_mask[0][0])):
            for key, value in colors.items():
                if predicted_mask[z][y][x][0] == key:
                    predicted_mask[z][y][x] = value
                    

for i in range(0,predicted_mask.shape[0]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(tf.keras.utils.array_to_img(X_test[i]), cmap='gray')
    ax2.imshow(tf.keras.utils.array_to_img(y_test[i].reshape(128,128,1), scale=True), cmap='gray')
    ax3.imshow(tf.keras.utils.array_to_img(predicted_mask[i]))
    
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    
    ax1.set_title('Real Image')
    ax2.set_title('Real Mask')
    ax3.set_title('Predicted Mask')
    
    plt.savefig(output_folder+f'comparison_{i}_.png', dpi=900, format='png')
    plt.close()
    #plt.show()
             
                