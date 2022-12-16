#%%
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os

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

x = [] # images
y = [] # categories 

print(type(x))
folder_burger = 'burger_pictures'
folder_box = 'boxes_pictures'

name_encoded = {"burger":0, "box":1 }

images_to_array(folder_box, "box")
images_to_array(folder_burger, "burger")

y = to_categorical(y, num_classes=2) #one hot encoding
x = np.array(x) # convert to numpy array 
show_image(0)
#Normalize data-> Now data is 0-255 , we need between -1 , 1 
x = ((x - 127.5)/127.5) 

show_image(90)

# %%
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential
# %%
# Create the Architecture for the CNN
model = Sequential()


model.add(Conv2D(32, (5,5), padding='same',activation='relu', input_shape=(100,100,3))) # we resize imagenes -> (100,100). rbg = 3
model.add(MaxPool2D(pool_size=(2,2))) # MaxPool -> max value pooling. we could use : min, average..

model.add(Conv2D(100, (5,5), padding='same',activation='relu')) 
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(100, (5,5), padding='same',activation='relu')) 
model.add(MaxPool2D(pool_size=(2,2)))



# %%
