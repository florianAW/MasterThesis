import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import sys
import seaborn as sns

import create_plots as plot
import settings as s
import process_image_data as process_data

df_data = process_data.load_image_data(resize=True, resize_dim=(s.DIM_X, s.DIM_Y), normalise=True, multiclass=s.MULTICLASS)

#save_test_images(df_data)

data_train, data_test = train_test_split(df_data, test_size=0.3, random_state=42)

X_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL).astype('float32') for img in data_train['img_vec']])
X_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL).astype('float32') for img in data_test['img_vec']])

y_train = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL).astype('int8') for img in data_train['mask_vec']])
y_test = np.array([img.numpy().reshape(s.DIM_X, s.DIM_Y, s.COLOR_CHANNEL).astype('int8') for img in data_test['mask_vec']])
y_test_flatten = y_test.flatten().astype('int8')

mean_px_variation = process_data.pixel_variance_per_label(X_train, y_train, show_plot=True)

classesIndexes, classesFrequency = np.unique(df_data['label'], return_counts=True)

if not s.MULTICLASS:
    
    ### load u-net model
    unet_model = keras.models.load_model(s.folder_saved_models+'best_model_binary.h5')
    
    #unet_model.summary()
    
    #score = unet_model.evaluate(X_test, y_test, verbose=1)
    
    predicted_mask = unet_model.predict(X_test)
    predicted_mask = tf.argmax(predicted_mask, axis=-1).numpy()
    predicted_mask = np.array([image.reshape(128,128,1) for image in predicted_mask])
    
    predicted_mask_flatten = predicted_mask.flatten().astype('int8')
    
    plot.Confusion_Matrix(y_truth=y_test_flatten, y_pred=predicted_mask_flatten, average_score=None, 
                          labels=['BG', 'Object'], save_plot=s.SAVE_PLOTS,
                          file_name='CM_Unet_binary')
    
    
    if s.EXPORT_MASK_IMAGES:
        for i in range(0,predicted_mask.shape[0]):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(tf.keras.utils.array_to_img(X_test[i]), cmap='gray')
            ax2.imshow(tf.keras.utils.array_to_img(y_test[i], scale=True), cmap='gray')
            ax3.imshow(tf.keras.utils.array_to_img(predicted_mask[i], scale=True), cmap='gray')
            
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            
            ax1.set_title('Real Image')
            ax2.set_title('Real Mask')
            ax3.set_title('Predicted Mask')
            
            plt.savefig(s.folder_pred_masks_binary+f'comparison_{i}_.png', dpi=900, format='png')
            plt.close()
            #plt.show()
    

else: ################################### multiclass

    ### load u-net model
    unet_model = keras.models.load_model(s.folder_saved_models+'best_model_C+P.h5')
    
    colors = {
                1: [255,   0,   0],
                2: [255, 128,   0],
                3: [255, 255,   0],
                4: [0,   255,   0],
                5: [0,     0, 255]
                }
    
    
    predicted_mask = unet_model.predict(X_test)
    predicted_mask = tf.argmax(predicted_mask, axis=-1).numpy()
    predicted_mask = np.array([image.reshape(128,128,1) for image in predicted_mask])
    
    predicted_mask_flatten = predicted_mask.flatten().astype('int8')  
    
    #plot.Confusion_Matrix(y_truth=y_test_flatten, y_pred=predicted_mask_flatten, average_score=None, 
    #                      labels=['BG', 'Type 1', 'Type 2', 'Type 3', 'Type 4', 'Crater'], save_plot=s.SAVE_PLOTS,
    #                      file_name='CM_Unet_multiclass')
    
    if s.EXPORT_MASK_IMAGES:
        predicted_mask_color = np.array([cv2.merge([img,img,img]) for img in predicted_mask])
        
        for z in range(len(predicted_mask_color)):
            for y in range(len(predicted_mask_color[0])):
                for x in range(len(predicted_mask_color[0][0])):
                    for key, value in colors.items():
                        if predicted_mask_color[z][y][x][0] == key:
                            predicted_mask_color[z][y][x] = value # Fill vector with respected color (e.g. Crater (5) = (0,0,255))
                            
        
        y_test_color = np.array([cv2.merge([img,img,img]) for img in y_test])
        
        for z in range(len(y_test_color)):
            for y in range(len(y_test_color[0])):
                for x in range(len(y_test_color[0][0])):
                    for key, value in colors.items():
                        if y_test_color[z][y][x][0] == key:
                            y_test_color[z][y][x] = value # Fill vector with respected color (e.g. Crater (5) = (0,0,255))
                            
        for i in range(0,predicted_mask.shape[0]):
            classesIndexes, classesFrequency = np.unique(predicted_mask[i], return_counts=True)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle(f'class frequency: {classesFrequency} \n {classesIndexes}')
            
            ax1.imshow(tf.keras.utils.array_to_img(X_test[i]), cmap='gray')
            ax2.imshow(tf.keras.utils.array_to_img(y_test_color[i]))
            ax3.imshow(tf.keras.utils.array_to_img(predicted_mask_color[i]))
            
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            
            ax1.set_title('Real Image')
            ax2.set_title('Real Mask')
            ax3.set_title('Predicted Mask')
            
            plt.savefig(s.folder_pred_masks_classes+f'comparison_{i}_.png', dpi=900, format='png')
            plt.close()
            #plt.show() 
    
    ################################################################# Post processing 
    ######################## Majority Voting
    
    
    mv_predicted_mask = process_data.pp_majority_vote(predicted_mask).astype('int8')
    
    mv_pred_mask_flatten = mv_predicted_mask.flatten()
    
    plot.Confusion_Matrix(y_truth=y_test_flatten, y_pred=mv_pred_mask_flatten, average_score=None, 
                          labels=['BG', 'Pit', 'Crater'], save_plot=s.SAVE_PLOTS,
                          file_name='CM_C+P_MV')
    
    predicted_labels = np.array([np.max(image) for image in mv_predicted_mask])
    
    plot.Confusion_Matrix(y_truth=data_test['label'], y_pred=predicted_labels, average_score=None, 
                          labels=['BG', 'Pit', 'Crater'], save_plot=s.SAVE_PLOTS,
                          file_name='CM_Unet_C+P_label_MV')
    
    if s.EXPORT_MASK_IMAGES:
        mv_predicted_mask_color = np.array([cv2.merge([img,img,img]) for img in mv_predicted_mask])
        
        for z in range(len(mv_predicted_mask_color)):
            for y in range(len(mv_predicted_mask_color[0])):
                for x in range(len(mv_predicted_mask_color[0][0])):
                    for key, value in colors.items():
                        if mv_predicted_mask_color[z][y][x][0] == key:
                            mv_predicted_mask_color[z][y][x] = value # Fill vector with respected color (e.g. Crater (5) = (0,0,255))
    
        
        for i in range(predicted_mask.shape[0]):
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            
            ax1.imshow(tf.keras.utils.array_to_img(X_test[i]), cmap='gray')
            ax2.imshow(tf.keras.utils.array_to_img(y_test_color[i]))
            ax3.imshow(tf.keras.utils.array_to_img(mv_predicted_mask_color[i]))
            
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            
            ax1.set_title('Real Image')
            ax2.set_title('Real Mask')
            ax3.set_title('Predicted Mask')
            
            plt.savefig(s.folder_pred_masks_classes_mv+f'comparison_{i}_.png', dpi=900, format='png')
            plt.close()
            #plt.show()
    
    
    ################################################################# Post processing 
    ######################## Pixel Variation
    
    pv_predicted_mask, px_variance = process_data.pp_classify_with_px_variance(X_test, predicted_mask, threshold=0.03).astype('int8')
                
    pv_pred_mask_flatten = pv_predicted_mask.flatten()
    
    plot.Confusion_Matrix(y_truth=y_test_flatten, y_pred=pv_pred_mask_flatten, average_score=None, 
                          labels=['BG', 'Type 1', 'Type 2', 'Type 3', 'Type 4', 'Crater'], save_plot=s.SAVE_PLOTS,
                          file_name='CM_Unet_multiclass_PV_03')
    
    predicted_labels_pv = np.array([np.max(image) for image in mv_predicted_mask])
    
    if s.EXPORT_MASK_IMAGES:
        pv_predicted_mask_color = np.array([cv2.merge([img,img,img]) for img in pv_predicted_mask])
        
        for z in range(len(pv_predicted_mask_color)):
            for y in range(len(pv_predicted_mask_color[0])):
                for x in range(len(pv_predicted_mask_color[0][0])):
                    for key, value in colors.items():
                        if pv_predicted_mask_color[z][y][x][0] == key:
                            pv_predicted_mask_color[z][y][x] = value # Fill vector with respected color (e.g. Crater (5) = (0,0,255))
            
        
        for i in range(predicted_mask.shape[0]):
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle(f'Pixel Variance: {px_variance[i]}')
            
            ax1.imshow(tf.keras.utils.array_to_img(X_test[i]), cmap='gray')
            ax2.imshow(tf.keras.utils.array_to_img(y_test_color[i]))
            ax3.imshow(tf.keras.utils.array_to_img(pv_predicted_mask_color[i]))
            
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            
            ax1.set_title('Real Image')
            ax2.set_title('Real Mask')
            ax3.set_title('Predicted Mask')
            
            plt.savefig(s.folder_pred_masks_classes+f'comparison_{i}_.png', dpi=900, format='png')
            plt.close()
            #plt.show()
    
    
    
                