import multiprocessing
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

from Preprocessing.Keras import extract_image_patches

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


def run_tensorflow():
    n = 10
    # images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100 in order
    images = [[[[ x * n + y + 1  for  z in range(2) ] for y in range(n+2)] for x in range(n)] ]



    startTime = datetime.now()
    gpu_fraction = 0.1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

    # result = session.run(sum_operation)
    images= np.asarray(images)
    print(images.shape, '\n\n')
    ps = tf.squeeze(
        extract_image_patches(images, ( 3, 3) , (1, 1),
                                 padding='same'), 0)
    print(ps)
    ps=tf.reshape(ps,[(images.shape[1]-2)*(images.shape[2]-2),-1 ])

    for pat in ps:
        print(pat)

    # print(tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
    #                                padding='VALID'), '\n\n')
    # print(tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
    #                                padding='VALID'), '\n\n')
    # print(list(tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
    #                                        padding='VALID').squeeze(0)[0] ), '\n\n')

            # print(tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 5, 5, 1], rates=[1, 2, 2, 1],
            #                                padding='VALID').eval(), '\n\n')
            # print(tf.extract_image_patches(images=images, ksizes=[1, 4, 4, 1], strides=[1, 7, 7, 1], rates=[1, 1, 1, 1],
            #                                padding='VALID').eval(), '\n\n')
            # print(tf.extract_image_patches(images=images, ksizes=[1, 4, 4, 1], strides=[1, 7, 7, 1], rates=[1, 1, 1, 1],
             #                              padding='SAME').eval())

    # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
    print("\n" * 5)
    print("Shape:", shape, "Device:", device_name)
    print("Time taken:", datetime.now() - startTime)

    print("\n" * 5)

if __name__ == "__main__":

    # option 1: execute code with extra process
    p = multiprocessing.Process(target=run_tensorflow)
    p.start()
    p.join()
