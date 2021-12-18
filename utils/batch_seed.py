import math
import operator
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def cos_batch(a, b):
    # obj= tf.sqrt(((a[None, :] - b[:, None]) ** 2))
    # return tf.math.reduce_sum(obj, axis=2)


    num = a @ b.T
    denom = tf.norm(a, axis=1).reshape(-1, 1) * tf.norm(b, axis=1)

    return num /denom

def get_weight(sim, bandwidth):
    thr = 1 - bandwidth

    if len(tf.config.experimental.list_physical_devices('DML'))>0:
        try:
            with tf.device('/DML:1'):
                max = tf.constant(1.0)
                min = tf.constant(0.0)

        except RuntimeError as e:
            print(e)
    else:
        max = tf.constant(1.0)
        min = tf.constant(0.0)

    dis = tf.where(sim > thr, max, min)

    return dis


def gaussian(dist, bandwidth):
    return tf.exp(-0.5 * ((dist / bandwidth)) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

def tensor_obj(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float64)
  return arg

def meanshift_torch(data, seed, bandwidth, max_iter=300):

    stop_thresh = 1e-3 * bandwidth
    iter = 0
    if len(tf.config.experimental.list_physical_devices('DML')) > 0:
        try:
            with tf.device('/DML:1'):
                X = tensor_obj(np.copy(data))
                S = tensor_obj(np.copy(seed))
                B = tf.constant(bandwidth, dtype=tf.float64)
        except RuntimeError as e:
            print(e)

    while True:
        weight = get_weight(cos_batch(S, X),B)

        num = (weight[:, :, None] * X).sum(dim=1)
        S_old = S
        S = num / weight.sum(1)[:,None]

        iter+=1

        if(tf.math.reduce_mean(tf.norm(S- S_old, axis=1)) < stop_thresh or iter == max_iter):
            break
    p_num=[]
    for line in weight:
        p_num.append(line[line==1].size()[0])

    my_mean = S.cpu().numpy()

    return my_mean, p_num

