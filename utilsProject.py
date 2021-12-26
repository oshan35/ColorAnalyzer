import cv2
from collections import Counter
import json
from PIL import Image
import json
import base64
from io import BytesIO
import numpy as np
from visualization import generate_pie_chart
import matplotlib.pyplot as plt

"""
    This file contains all the util funtions. Image resizing pre processing work post processing work and
    color space converters. There few unused methods in this file which will need in the future.
"""


def resize(image,segment):
    image = cv2.resize(image, (200, 300), interpolation=cv2.INTER_AREA)
    image_data = image.reshape(image.shape[0] * image.shape[1], 3)
    if segment: image_data = np.array(list(filter(lambda a: a != [0, 0, 0], image_data.tolist())))
    return image_data


def convert_BGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def convert_LAB2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)


def pre_process(img_original,segment):
    """
    Work using the RGB color space in your code, where Euclidean distances in this space correlate
    better with color changes perceived by the human eye.
    """

    # img_original = cv2.imread(PATH)
    img= img_original
    #if segment: img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # plt.imshow(img)
    img_post = resize(img,segment)
    return img_post

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def post_processing(image,cluster, labels):

    """

    :param cluster: Meanshift object
    :param labels: cluster labels for each data point
    :param segment: boolean which tells if this is a segmented image. If so we have to remove the dead pixels
    :return: Json object with the analyzed data

    This method calculates the color percentages from extracted
    """
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = cluster.cluster_centers_

    center_colors = center_colors.tolist()

    ordered_colors = [center_colors[i] for i in counts.keys()]

    rgb_colors = [list(map(int,ordered_colors[i])) for i in counts.keys()]

    colors_data = {}
    for color in range(len(rgb_colors)):
        percentage = round((counts[color] / (sum(counts.values()))) * 100, 2)
        colors_data[str(rgb_colors[color])] = percentage

    # Convert data color details into json object
    color_data_json = json.dumps(colors_data)

    generate_pie_chart(counts, rgb_colors, [RGB2HEX(ordered_colors[i]) for i in counts.keys()], image)

    # Returning analyzed data as a json object
    return color_data_json


def convert_json_to_image(json_object):
    """
    :param json_object:
    :return:
    """
    dict_data = json.load(json_object)

    image = dict_data["img"]
    image = base64.b64decode(image)
    image = BytesIO(image)
    return image


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)
