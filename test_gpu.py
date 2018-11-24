import multiprocessing
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime



def run_tensorflow():
    device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
    shape = (int(sys.argv[2]), int(sys.argv[2]))
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    with tf.device(device_name):
        random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)


    startTime = datetime.now()
    gpu_fraction = 0.1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)


    with tf.Session(config=tf.ConfigProto(log_device_placement=True,gpu_options=gpu_options)) as session:
            result = session.run(sum_operation)
            print(result)

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

    # wait until user presses enter key
    input()

    # option 2: just execute the function
    run_tensorflow()

    # wait until user presses enter key
    input()