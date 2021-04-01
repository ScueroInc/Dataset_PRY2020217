from subprocess import call
import os
from urllib.request import urlretrieve
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


    

    
def load_data(model):
    img_size = 0
    if(model=="Resnet"):
        img_size=224
    else:
        img_size=299
        
    print(img_size)


        
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    class_names = []
    category_num = 0
    
    for category in sorted(os.listdir("Images")):
        class_names.append(category)
        count = 0
        for img in sorted(os.listdir(os.path.join("Images", category))):
                
            x = load_img(os.path.join("Images", category, img),target_size=(img_size,img_size))
            if count < 40:
                x_test.append(img_to_array(x))
                y_test.append(category_num)
            else:
                x_train.append(img_to_array(x))
                y_train.append(category_num)
            count += 1
        category_num += 1
    print("Entrenamiento:",len(np.array(x_train)))
    print("Test:",len(np.array(x_test)))
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test)), class_names
    
