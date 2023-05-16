# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:08:33 2023

@author: flori
"""

''' How to design an algo '''
'''
    XX - YY - ZZ
    
    XX: Which deep learning architecture 
    YY: Binary or multiclass problem (B / MC)
    ZZ: Which type of object will be classified
'''
import os


def set_algo_id(algo):
    
    available_algos = [
                        'Unet - B - objects',
                        'Unet - MC - Pit Types + Crater',
                        'Unet - MC - C/P'
                      ]
    global algo_id

    if algo in available_algos:
        algo_id = algo
    else:
        raise Exception('Algo ID not available, please set another ID')
 
SAVE_PLOTS = True
EXPORT_MASK_IMAGES = False

MULTICLASS, NUMBER_CLASSES = True, 6
DIM_X, DIM_Y, COLOR_CHANNEL = 512, 512, 1

folder_plots = 'plots/'
folder_saved_models = 'saved_models/'

folder_pits = 'data/pits/'
folder_mask_classes = 'data/pit_masks_classes/'
folder_mask_binary = 'data/pit_masks_binary/'
folder_mask_CP = 'data/pit_masks_C+P/'

folder_pred_masks_binary = 'data/predicted_masks_binary/'
folder_pred_masks_CP = 'data/predicted_masks_C+P/'
folder_pred_masks_classes = 'data/predicted_masks_classes/'
folder_pred_masks_classes_mv = 'data/predicted_masks_classes_mv/'
folder_pred_masks_classes_pv = 'data/predicted_masks_classes_pc/'








        

        

