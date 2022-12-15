#%%
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os

x = [] # images
y = [] # categories 

print(type(x))
folder_burger = 'burger_pictures'
folder_box = 'boxes_pictures'

name_encoded = {"burger":0, "box":1 }

def images_to_array(folder,name):
    for imgage in os.listdir(folder):
        loaded_image = Image.open(os.path.join(folder, imgage))
        resize_image = Image.Image.resize(loaded_image, [100,100])
        image_array = np.array(resize_image)
        x.append(image_array) #images
        y.append(name_encoded[name]) #categories


def show_image(index):
    plt.imshow(x[index])
    plt.show()
    print(y[index])

images_to_array(folder_box, "box")
images_to_array(folder_burger, "burger")

y = to_categorical(y, num_classes=2) #one hot encoding

show_image(0)
#%%