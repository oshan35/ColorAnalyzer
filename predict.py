import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
# from train import create_dir
import matplotlib.pyplot as plt

""" Global parameters """
H = 512
W = 512


def segmentation(image_dict, save_extracted_object=False):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    # create_dir("test_images/mask")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("deeplab-model/model.h5")

    """ Load the dataset """
    data_x = image_dict

    predictions = {}

    for name, image in tqdm(data_x.items(), total=len(data_x)):
        """ Extracting name """
        name = name

        """ Reading the image """
        # image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        """ Save the image """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)

        if save_extracted_object:
            cv2.imwrite(f"sample-preds/{name}.png", cat_images)

        predictions[name] = cat_images

    return predictions
