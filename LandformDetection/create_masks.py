# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:20:20 2022

@author: flori
"""

import pandas as pd
import numpy as np
import os
import json
import cv2
import collections

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

MULTICLASS = True

import_path = 'data/pits/'
if MULTICLASS:
    export_path = 'data/pit_masks_classes/'
else:    
    export_path = 'data/pit_masks/'

json_files= [file for file in os.listdir(import_path) if file.endswith('.json')]

classes = {
            1: ['Type-1', 'Type-1a', 'Type-1b'],
            2: ['Type-2', 'Type-2a', 'Type-2b'],
            3: ['Type-3'],
            4: ['Type-4'],
            5: ['Crater']
            }

masks = {}
labels = []
for file_name in json_files:
    
    f = open(import_path+file_name)
    data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    
    mask = np.zeros((width, height))
    
    try:
        polygon_pts = np.array(data['shapes'][0]['points'], dtype=np.int32)
        label = data['shapes'][0]['label']
        labels.append(label)
        for key, value in classes.items():
            if label in value:
                int_label = key
        #int_label = 2 if label == 'Crater' else 1 
    
    except:
        print(file_name)
    
    if MULTICLASS:
        cv2.fillPoly(mask, [polygon_pts], color=(int_label))
    else:
        cv2.fillPoly(mask, [polygon_pts], color=(255))
    
    masks[file_name[:-5]] = mask
    
    cv2.imwrite(export_path+file_name[:-5]+'.png', mask)

    f.close()
    

counter = collections.Counter(labels)

