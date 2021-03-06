#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 
# ### Download Images
import argparse
import multiprocessing
import os
from os.path import isfile, isdir
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing.utils import load_noise_img, load_img, imshow, show_results

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

max_size=260

def load_and_process_img(path_to_img):
    img = load_img(path_to_img,max_size)
    print(img.shape)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

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

class Gatyes:

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    results = "gatyes_results/"

    def __init__(self,device_name,iter=1000,dataset_folder_path='final_content/'):
        self.device_name=device_name
        self.iterations=iter

        if not os.path.exists(self.results + dataset_folder_path):
            os.makedirs(self.results + dataset_folder_path)

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

    def gram_matrix(self,input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        #   print(channels)
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        #   print(a)
        #   print(n)
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)


    def get_style_loss(self,base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.get_shape().as_list()
        gram_style = self.gram_matrix(base_style)

        return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


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
        return style_features, content_features


    # ### Computing the loss and gradients
    def compute_loss(self,model, loss_weights, init_image, gram_style_features, content_features):
        """This function will compute the loss total loss.

        Arguments:
          model: The model that will give us access to the intermediate layers
          loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
          content_features: Precomputed outputs from defined content layers of
            interest.

        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = model(init_image)

        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_per_style_layer = 1.0 / float(self.num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score


    def compute_grads(self,cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    # ### Optimization loop

    def run_style_transfer(self,content_path,
                           style_path,
                           num_iterations=1000,
                           content_weight=1e3,
                           style_weight=1e-2):

        # We don't need to (or want to) train any layers of our model, so we set their
        # trainable to false.
        model = self.get_model()
        for layer in model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = self.get_feature_representations(model, content_path, style_path)
        gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]

        # Set initial image
        init_image = load_and_process_img(content_path)
        init_image = tfe.Variable(init_image, dtype=tf.float32)

        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        # For displaying intermediate images
        iter_count = 1

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iterations / (num_rows * num_cols)
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        imgs = []

        for i in range(num_iterations):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])


            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end_time = time.time()

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())

            if i % 1 == 0:
                start_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = deprocess_img(plot_img)

                if i% display_interval==0:
                    imgs.append(plot_img)

                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))


        print('Total time: {:.4f}s'.format(time.time() - global_start))
        plt.figure(figsize=(14, 4))

        for i, img in enumerate(imgs):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.draw()

        plt.savefig(self.results + content_path + '_inter.jpg')
        return best_img, best_loss

    def run_tensorflow(self,content_path,style_path):

        with tf.device(self.device_name):
            best, best_loss = self.run_style_transfer(content_path,style_path, num_iterations=self.iterations)
            show_results(self.results,best,content_path,style_path)

    def run_tensorflow2(self):
        ## running for our dataset
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

        print (Images)
        np.random.shuffle(Images)
        np.random.shuffle(sImages)
        print (sImages)
        Style_Images, Content_Images = sImages , Images

        no_images=100;

        with tf.device(self.device_name):
            i=0;
            print("fadsf")
            for style_path ,content_path in zip(Style_Images, Content_Images):
                print(content_path, style_path)
                content_path = dataset_folder_path + "/" + content_path;
                style_path = sdataset_folder_path + "/" + style_path;

                best, best_loss = self.run_style_transfer(content_path, style_path, num_iterations=self.iterations)
                show_results(self.results,best, content_path, style_path)
                if i==no_images:
                    break;
                i+=1

#
if __name__ == "__main__":

    device_name = "/gpu:0"
    content_path = 'samples/ck.jpg'
    content_map_path = 'samples/ck_color_mask.png'
    style_path = 'samples/Renoir.jpg'
    style_map_path = 'samples/Renoir_color_mask.png'
    parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument
    add_arg('--content', default=None, type=str)
    add_arg('--style', default=None, type=str)
    add_arg('--output', default='output.png', type=str)
    add_arg('--output-size', default=None, type=str)
    add_arg('--iterations', default=100, type=int)
    add_arg('--device', default='cpu', type=str)
    add_arg('--model', default='Gateys', type=str)
    add_arg('--folder', default='samples', type=str)

    args = parser.parse_args()

    device_name = args.device
    content_path = args.content
    style_path = args.style
    folder = args.folder
    iter = args.iterations


    obj = Gatyes(device_name, iter, folder);

    p = multiprocessing.Process(target=obj.run_tensorflow(content_path,style_path))
    #p = multiprocessing.Process(target=obj.run_tensorflow2())
    p.start()
    p.join()
    plt.show()

