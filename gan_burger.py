# %%
from keras.layers import *
from keras.models import Sequential, Model
import glob  # uploading files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# %%

# loading images into an array -> to input in the model
image_path = 'burger_pictures/*'
images_array = []

for image in glob.glob(image_path):
    img = Image.open(image)
    img = Image.Image.resize(img, (240, 240))
    img = np.array(img)
    images_array.append(img)

images_array = np.array(images_array)

X_train = images_array.astype('float32')
X_train = (X_train - 127.5)/127.5

# %%
# %%
generator = Sequential([
    Dense(128*60*60, input_dim=100, activation=LeakyReLU(0.2)),
    BatchNormalization(),
    Reshape((60, 60, 128)),
    UpSampling2D(),
    Conv2D(128, 5, 5, padding='same', activation=LeakyReLU(0.2)),
    BatchNormalization(),
    Conv2D(56, 3, 3, padding='same', activation=LeakyReLU(0.2)),
    BatchNormalization(),
    UpSampling2D(),
    Conv2D(3, 5, 5, padding='same', activation='tanh')
])
generator.summary()
#%%
discriminator = Sequential([
    Conv2D(124, (5, 5), strides=2, input_shape=(240, 240, 3),
           padding='same', activation=LeakyReLU(0.2)),
    Dropout(0.4),
    Conv2D(124, (5, 5), strides=2, padding='same', activation=LeakyReLU(0.2)),
    Dropout(0.4),
    Flatten(),
    Dense(1, activation='sigmoid')
])
discriminator.summary()
# %%
