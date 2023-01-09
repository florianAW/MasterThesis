
import numpy as np
import pandas as pd
import os
import cv2
import time
import tensorflow as tf

def load_image_data(resize:bool, resize_dim:tuple, normalise:bool, multiclass:bool) -> pd.DataFrame:
    start=time.time()
    
    folder_to_img = 'data/pits/'
    if multiclass:
        folder_to_masks = 'data/pit_masks_classes/'
    else:
        folder_to_masks = 'data/pit_masks/'
    
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
    
    print('Data loaded...')
    return df_total

def save_test_images(df_data):
    for i in range(len(df_data)):
        cv2.imwrite('data/resized_pits/'+df_data.index[i]+'.png', df_data['img_vec'][i])
        cv2.imwrite('data/resized_masks/'+df_data.index[i]+'.png', df_data['mask_vec'][i])
    
    print('Images saved...')
    



