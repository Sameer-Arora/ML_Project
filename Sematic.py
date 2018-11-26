#!/usr/bin/env python
# coding: utf-8
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

# In[2]:


import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

from PIL import Image
import time
import functools

import tensorflow as tf
import tensornets as nets
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

from Preprocessing import utils
from Preprocessing.utils import load_data, load_img, imshow, load_noise_img, show_results
from Preprocessing import Theano

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

dataset_folder_path = 'train_1'
utils.load_data(dataset_folder_path)


### Prepare the data
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    # print("img shape",img.shape)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


##Function to get shape for total variation loss
# def _tensor_size(tensor):
#     from operator import mul
#     return reduce(mul, (d.value for d in tensor.get_shape()), 1)
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

class Semantic():
    # Content layer where will pull our feature maps
    content_layers = ['block4_conv2']

    # Style layer we are interested in
    style_layers = ['block3_conv1',
                    'block4_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)


    def __init__(self,device_name):
        self.device_name=device_name

    # ## Build the Model
    def get_model(self):
        """ Creates our model with access to intermediate layers.

        This function will load the VGG19 model and access the intermediate layers.
        These layers will then be used to create a new model that will take input image
        and return the outputs from these intermediate layers from the VGG model.

        Returns:
          returns a keras model that takes image inputs and outputs the style and
            content intermediate layers.
        """
        # Load our model. We load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        # Build model
        return models.Model(vgg.input, model_outputs)


    # ### Content Loss
    def get_content_loss(self,base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))


    ## Style Loss
    def get_style_loss_(self,base_style_patches, target_style):
        """Returns a list of loss components as Theano expressions. Finds the best style patch for each patch in the
        current image using normalized cross-correlation, then computes the mean squared error for all patches.
        """
        style_loss = 0
        # print(target_style)
        # Extract the patches from the current image, as well as their magnitude.
        size = 3
        stride = 4
        patches = tf.squeeze(
            tf.extract_image_patches(target_style, ksizes=[1, size, size, 1], strides=[1, stride, stride, 1]
                                     , rates=[1, 1, 1, 1], padding='VALID'), 0)

        #    patches=tf.reshape(patches,[(target_style.shape[1]-2)*(target_style.shape[2]-2),-1 ])
        patches = tf.reshape(patches, [
            ((target_style.shape[1] - size) // stride + 1) * ((target_style.shape[2] - size) // stride + 1), -1])
        # print(patches)

        for patch in patches:
            min_norm = 10000000000;
            sel_nei = 0;

            for base_patch in base_style_patches:
                # print(patch)
                # print(base_patch)
                norm = tf.reduce_sum(tf.multiply(patch, base_patch))
                cross_corre = norm / (tf.norm(patch) * tf.norm(base_patch));
                if min_norm > cross_corre:
                    min_norm = cross_corre;
                    sel_nei = tf.convert_to_tensor(base_patch);

            style_loss += tf.reduce_mean(tf.square(patch - sel_nei));

        return style_loss


    # ## Apply style transfer to our images
    # ### Run Gradient Descent
    def get_feature_representations(self,model, content_path, style_path):
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess both the content and style
        images from their path. Then it will feed them through the network to obtain
        the outputs of the intermediate layers.

        Arguments:
          model: The model that we are using.
          content_path: The path to the content image.
          style_path: The path to the style image

        Returns:
          returns the style features and the content features.
        """
        # Load our images in
        content_image = load_and_process_img(content_path)
        style_image = load_and_process_img(style_path)

        # batch compute content and style features
        style_outputs = model(style_image)
        content_outputs = model(content_image)

        # Get the style and content feature representations from our model
        style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
        # print("NO style ",num_style_layers)
        # print("Base featurs",content_features)
        return style_features, content_features


    def get_transVar_loss(self,image):
        #     tv_y_size = _tensor_size(image[:, 1:, :, :])
        #     tv_x_size = _tensor_size(image[:, :, 1:, :])
        #     return 2 * ((tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :image_content.shape[1] - 1, :, :]) / tv_y_size) + (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :image_content.shape[2] - 1, :]) /tv_x_size))
        loss = tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])) + \
               tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        return loss


    # ### Computing the loss and gradients
    def compute_loss(self,model, loss_weights, init_image, base_style_patches, content_features,content_map_features):
        """This function will compute the loss total loss.
        Arguments:
          model: The model that will give us access to the intermediate layers
          loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          base_style_features: Precomputed style patches corresponding to the
            defined style layers of interest.
          content_features: Precomputed outputs from defined content layers of
            interest.
        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight, trans_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = model(init_image)

        print("NO style ", self.num_style_layers)
        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]
        style_output_features_new=[]

        for style_feat_img,content_map_img in zip(style_output_features,content_map_features):
            print(style_feat_img.shape)
            content_map_img=tf.expand_dims(content_map_img,axis=0)
            print(content_map_img.shape)
            style_feat_img= tf.concat([ style_feat_img,content_map_img],-1 )
            print(style_feat_img.shape)
            style_output_features_new.append(style_feat_img)

        style_output_features=style_output_features_new

        #
        # for base in content_output_features:
        #     print("gen- shape",base[0].shape)
        #
        # for base in content_features:
        #     print("inp-shape",base.shape)

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        weight_per_style_layer = 1.0 / float(self.num_style_layers)

        for base_sty_patch, target_style in zip(base_style_patches, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss_(base_sty_patch, target_style)

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content[0])

        trans_score = self.get_transVar_loss(init_image)
        trans_score *= trans_weight
        style_score *= style_weight
        content_score *= content_weight

        # Get total loss
        loss = style_score + content_score + trans_score
        return loss, style_score, content_score, trans_score


    def compute_grads(self,cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss


    # ### Optimization loop
    import IPython.display


    def run_style_transfer(self,content_path, style_path,
                           content_map_path="", style_map_path="",
                           num_iterations=1000,
                           content_weight=10,
                           style_weight=25, trans_weight=1):
        # We don't need to (or want to) train any layers of our model, so we set their
        # trainable to false.
        model = self.get_model()
        for layer in model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = self.get_feature_representations(model, content_path, style_path)
        style_map_features,_ = self.get_feature_representations(model, content_map_path, style_map_path)
        content_map_features,_ = self.get_feature_representations(model,  style_map_path,content_map_path)

        size = 3
        stride = 4
        base_style_patches = []
        i = 0
        style_img = load_img(style_path)

        for style_feat_img,style_map_img in zip(style_features,style_map_features):
            print(style_feat_img.shape)
            print(style_map_img.shape)
            style_feat_img= tf.concat([ style_feat_img,style_map_img],-1 )
            print(style_feat_img.shape)

            li = tf.squeeze(tf.extract_image_patches(tf.expand_dims(style_feat_img, axis=0), ksizes=[1, size, size, 1],
                                                     strides=[1, stride, stride, 1]
                                                     , rates=[1, 1, 1, 1], padding='VALID'), 0)
            li = tf.reshape(li, [
                ((style_feat_img.shape[0] - size) // stride + 1) * ((style_feat_img.shape[1] - size) // stride + 1), -1])
            # li = tf.reshape(li, [(style_feat_img.shape[0] - 2) * (style_feat_img.shape[1] - 2), -1])
            base_style_patches.append(li)

            # print( i,len( base_style_patches[i] ), base_style_patches[i][0] )
            i += 1
        # print(len(base_style_patches))

        # Set initial image
        init_image = load_noise_img(load_and_process_img(content_path))
        init_image = load_and_process_img(content_path)

        init_image = tfe.Variable(init_image, dtype=tf.float32)
        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        # For displaying intermediate images
        iter_count = 1

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (style_weight, content_weight, trans_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'base_style_patches': base_style_patches,
            'content_features': content_features,
            'content_map_features': content_map_features
        }

        # For displaying
        num_rows = 10
        num_cols = 100
        display_interval = 1
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        imgs = []
        for i in range(num_iterations):
            print("himmat rakho")
            grads, all_loss = self.compute_grads(cfg)
            print("gradient aega")
            loss, style_score, content_score, trans_score = all_loss
            opt.apply_gradients([(grads, init_image)])

            print("gradient agya")
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            # print("II 1",init_image)
            init_image.assign(clipped)
            # print("II 2",cfg['init_image'])
            end_time = time.time()

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())

            if i % display_interval == 0:
                start_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = deprocess_img(plot_img)
                imgs.append(plot_img)
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'trans loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score, trans_score, time.time() - start_time))

        print('Total time: {:.4f}s'.format(time.time() - global_start))
        plt.figure(figsize=(14, 4))
        for i, img in enumerate(imgs):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        return best_img, best_loss

    def run_tensorflow(self,content_path,style_path):
        with tf.device("/gpu:0"):
            imshow(deprocess_img(load_and_process_img(content_path)), squeze=False)
            plt.show()
            best, best_loss = self.run_style_transfer(content_path, style_path,
                                                 style_map_path, content_map_path, num_iterations=15)
            show_results(results, best, content_path, style_path)


    ## Run for complete dataset.
    def run_tensorflow2(self):
        ## running for our dataset
        Images = []
        sImages = []

        dataset_folder_path = 'final_content'
        sdataset_folder_path = 'style'

        for filename in os.listdir(dataset_folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                Images.append(filename)

        for filename in os.listdir(sdataset_folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                sImages.append(filename)

        # print (Images)
        np.random.shuffle(Images)
        np.random.shuffle(sImages)
        # print (Images)
        Style_Images, Content_Images = sImages[: round(0.50 * len(sImages))], Images[round(0.50 * len(Images)):]
        no_images = 100;

        with tf.device("/cpu:0"):

            i = 0;
            for style_path ,content_path in zip(Style_Images, Content_Images):
                print("Path", content_path, style_path)

                content_path = dataset_folder_path + "/" + content_path;
                style_path = sdataset_folder_path + "/" + style_path;
                content_map_path = content_path.split('.')[0] + "_sem.jpg";
                style_map_path = style_path.split('.')[0] + "_sem.jpg";

                best, best_loss = self.run_style_transfer(content_path, style_path,content_map_path=content_map_path,style_map_path=style_map_path, num_iterations=25)

                show_results(results, best, content_path, style_path)
                if i == no_images:
                    break
                i += 1


results = "results/"
dataset_folder_path = 'samples/'

if not os.path.exists(results + dataset_folder_path):
    os.makedirs(results + dataset_folder_path)


if __name__ == "__main__":

    device_name = "/gpu:0"

    content_path = 'samples/Freddie.jpg'
    content_map_path = 'samples/Freddie_color_mask.png'
    style_path = 'samples/Mia.jpg'
    style_map_path = 'samples/Mia_color_mask.png'

    obj = Semantic(device_name);
    p = multiprocessing.Process(target=obj.run_tensorflow(content_path,style_path))
    p.start()
    p.join()

    # option 2: just execute the function for whole dataset
    # p = multiprocessing.Process(target=run_tensorflow2)
    # p.start()
    # p.join()

    # wait until user presses enter key
    plt.pause(10)