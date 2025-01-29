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
        break
    break
img_size = 244
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()

traning_data = []
def create_training_data():
    for category in classes:
        path = os.path.join(data_directory,cateogry)
        classes_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                traning_data.append([new_array,classes_num])
            except Exception as e:
                pass     