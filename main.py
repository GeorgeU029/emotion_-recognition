import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

data_directory = "Training/"
classes = ["0","1","2","3","4","5","6"]

for cateogry in classes:
    path = os.path.join(data_directory,cateogry)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()