
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import settings as s

def Confusion_Matrix(y_truth, y_pred, average_score, labels, save_plot, file_name):
    
    cm = confusion_matrix(y_truth, y_pred)
    
    f1 = f1_score(y_truth, y_pred, average=average_score)
    presicion = precision_score(y_truth, y_pred, average=average_score)
    recall = recall_score(y_truth, y_pred, average=average_score)
    
    f1 = [round(val, 2) for val in f1]
    presicion = [round(val, 2) for val in presicion]
    recall = [round(val, 2) for val in recall]
    
    plt.figure(figsize=(9,6), dpi=1200)
    ax = plt.axes()
    plt.title(f'Confusion Matrix \nPrecision: {presicion}\n Sensitivity: {recall}')
    
    sns.heatmap(cm, annot=True)
    
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    plt.tight_layout()
    if s.SAVE_PLOTS:
        plt.savefig(s.folder_plots+file_name+'.png', format='png')

