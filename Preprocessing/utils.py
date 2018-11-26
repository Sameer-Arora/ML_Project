# ## Setup
import multiprocessing
import os
from os.path import isfile, isdir
import numpy as np
import matplotlib.pyplot as plt

import theano
from IPython import get_ipython
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread

# ### Import and configure modules

import matplotlib as mpl
from skimage.transform import resize

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

from PIL import Image
import time
import functools

# In[3]:
import tensorflow as tf
import tensornets as nets
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


def load_data(dataset_folder_path):
    ## loading the batch data for the style transfer
    import zipfile
    from zipfile import ZipFile

    if not isdir(dataset_folder_path):
        with ZipFile('train_1.zip') as tar:
            tar.extractall()

            tar.close()


def load_noise_img(img):
    img = np.random.random_integers(0, high=255, size=img.shape)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def load_img(path_to_img,max_dim=260):

    img = Image.open(path_to_img).convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((max_dim,max_dim), Image.ANTIALIAS)
    #img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    #img = img.resize((224, 224), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


# def load_img(path_to_img):
#     max_dim = 512
#     img = imread(path_to_img,plugin="matplotlib")
#     ## converting grayscale to RGB
#     if len(img.shape)==2:
#         img=gray2rgb(img);
#
#     long = max(img.shape)
#     scale = max_dim / long
#
#     img = resize(img,(224,224), anti_aliasing=True)
#     img = kp_image.img_to_array(img)
#     # We need to broadcast the image array such that it has a batch dimension
#     img = np.expand_dims(img, axis=0)
#     return img

def imshow(img, title=None, squeze=True):
    if squeze:
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
    else:
        out = img
    # Normalize for display
    out = out
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def show_results(results,best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path).astype('uint8')
    style = load_img(style_path).astype('uint8')

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.savefig(results + content_path + '.jpg')

    if show_large_final:
        plt.figure(figsize=(10, 10))
        plt.imshow(best_img)
        plt.title('Output Image')
        plt.savefig(results + content_path + '_res.jpg')
        #plt.show()
