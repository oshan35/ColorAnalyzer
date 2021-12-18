import tensorflow as tf

import numpy as np

# xs = tf.Variable(np.array([[1, 2],[3, 4]]), dtype = tf.float32)
# l1 = tf.norm(xs, axis = 1)
# l1  = tf.math.reduce_mean(l1, axis= None)
# print(l1)

# Place tensors on the CPU
# with tf.device('/DML:1'):
#   a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#   b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#
# # Run on the GPU
# c = tf.matmul(a, b)
# print(c)
# print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('DML')))
