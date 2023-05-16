import numpy as np
import pandas as pd
import os
import cv2
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import settings as s

def load_image_data(resize:bool, resize_dim:tuple, normalise:bool, multiclass:bool) -> pd.DataFrame:
    start=time.time()
    
    folder_to_img = s.folder_pits
    if multiclass:
        folder_to_masks = s.folder_mask_classes
    else:
        folder_to_masks = s.folder_mask_binary
    
    img_data = {
                'file_name':[],
                'img_vec':[]
                }
    
    mask_data = {
                'file_name':[],
                'mask_vec':[]
                }
    
    img_files = [file for file in os.listdir(folder_to_img) if file.endswith('.tiff')]
    mask_files = [file for file in os.listdir(folder_to_masks) if file.endswith('.png')]
    
    
    for img_file in img_files:
        file_name = img_file[:-5]
        img_vec = cv2.imread(folder_to_img+img_file, 0)
        if resize:
            img_vec = cv2.resize(img_vec, resize_dim, interpolation = cv2.INTER_LINEAR)
        if normalise:
            img_vec = tf.cast(img_vec, tf.float32) / 255.0
        
        img_data['file_name'].append(file_name)
        img_data['img_vec'].append(img_vec)
        
    for mask_file in mask_files:
        file_name = mask_file[:-4]
        mask_vec = cv2.imread(folder_to_masks+mask_file, 0)
        if resize:
            mask_vec = cv2.resize(mask_vec, resize_dim, interpolation = cv2.INTER_LINEAR)
        if normalise and not multiclass:
            mask_vec = tf.cast(tf.cast(mask_vec, tf.float32) / 255.0, tf.int8)
        else:
            mask_vec = tf.cast(mask_vec, tf.int8)
        
        mask_data['file_name'].append(file_name)
        mask_data['mask_vec'].append(mask_vec)
    
    df_img = pd.DataFrame.from_dict(img_data)
    df_mask = pd.DataFrame.from_dict(mask_data)
    
    df_total = df_img.set_index('file_name').join(df_mask.set_index('file_name'), how='inner')
    
    df_total['label'] = np.array([np.max(df_total['mask_vec'][i]) for i in range(len(df_total))])
    
    df_total['px_variance'] = [calc_px_variance(image.numpy(), mask.numpy()) for image, mask in zip(df_total['img_vec'], df_total['mask_vec'])]
    
    print('Data loaded...')
    return df_total

def save_test_images(df_data):
    for i in range(len(df_data)):
        cv2.imwrite('data/resized_pits/'+df_data.index[i]+'.png', df_data['img_vec'][i])
        cv2.imwrite('data/resized_masks/'+df_data.index[i]+'.png', df_data['mask_vec'][i])
    
    print('Images saved...')
    
def pp_majority_vote(pred_masks):
      
    #loop through every img. Get the most predicted label (if there are at least 3 label). 
    #Replace every pixel except the background with the most common label
    
    for z in range(len(pred_masks)): 
        classesIndexes, classesFrequency = np.unique(pred_masks[z], return_counts=True)
        
        if len(classesFrequency) <= 2:
            continue
        else:
            most_common_pred = np.argmax(classesFrequency[1:]) + 1 # ignore background label
        
        for y in range(len(pred_masks[0])):
            for x in range(len(pred_masks[0][0])):
                if pred_masks[z][y][x][0] != 0:
                    pred_masks[z][y][x][0] = most_common_pred
                    
    return pred_masks

def pp_classify_with_px_variance(X_test, pred_masks, threshold):
    px_variance = []
    
    for z in range(pred_masks.shape[0]): # Look which predictions occur in every img
        classesIndexes, classesFrequency = np.unique(pred_masks[z], return_counts=True)
        
        px_variation = calc_px_variance(X_test[z], pred_masks[z])
        px_variance.append(px_variation)
        
        if 2 in classesIndexes and 4 in classesIndexes: # if Type 2 and 4 then use this classification method
    
            if px_variation > threshold:
                prediction = 2
            else:
                prediction = 4
                
            for y in range(len(pred_masks[0])):
                for x in range(len(pred_masks[0][0])):
                    if pred_masks[z][y][x][0] != 0:
                        pred_masks[z][y][x][0] = prediction
                        
    return pred_masks, px_variance
    
def pixel_variance_per_label(X_train, y_train, show_plot): # only works for only one label per image
    
    masked_images = {
        'label' : [],
        'px_variation' : []
        }
    
    for i in range(len(X_train)):
        masked_img = X_train[i].copy()
        masked_img[y_train[i] == 0] = -1
        masked_img[y_train[i] != 0] = X_train[i][y_train[i] != 0]
        
        masked_img = masked_img.flatten()
        masked_img = masked_img[masked_img != -1]
        
        px_variation = np.var(masked_img)
        
        label = np.max(y_train[i])
        
        masked_images['label'].append(label)
        masked_images['px_variation'].append(px_variation)
        
    df_masked_images = pd.DataFrame.from_dict(masked_images)
    
    mean_px_variation = df_masked_images.groupby('label').mean()
    
    if show_plot:
        plt.hist(df_masked_images[df_masked_images['label'] == 2]['px_variation'], 
                 bins=30, color=(1,0.5,0,0.5), label='Type-2 Pit')
        plt.hist(df_masked_images[df_masked_images['label'] == 4]['px_variation'], 
                 bins=30, color=(0,1,0,0.5), label='Type-4 Pit')
        plt.vlines(0.03, 0, 60, (1,0,0,1), label='Threshold: 0.03')
        plt.legend()
    
    return mean_px_variation

    
def calc_px_variance(image, mask):
    masked_img = image.copy()
    masked_img[mask == 0] = -1
    masked_img[mask != 0] = image[mask != 0]
    
    masked_img = masked_img.flatten()
    masked_img = masked_img[masked_img != -1]
    
    px_variation = np.var(masked_img)
    
    return px_variation