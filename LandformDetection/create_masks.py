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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


import_path = 'data/pits/'
export_path = 'data/pit_masks/'

json_files= [file for file in os.listdir(import_path) if file.endswith('.json')]

masks = {}
for file_name in json_files:
    
    f = open(import_path+file_name)
    data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    
    mask = np.zeros((width, height))
    
    try:
        polygon_pts = np.array(data['shapes'][0]['points'], dtype=np.int32)
    except:
        print(file_name)
        
    cv2.fillPoly(mask, [polygon_pts], color=(255))
    
    masks[file_name[:-5]] = mask
    
    cv2.imwrite(export_path+file_name[:-5]+'.png', mask)

    f.close()
    


