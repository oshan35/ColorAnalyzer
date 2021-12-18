import time

from clustering import meanshift
from clustering import sillhouttle_analysis
from objectSegmentation import seperate_objs_from_image
from utilsProject import pre_process
from utilsProject import post_processing
from PIL import Image
from utilsProject import convert_json_to_image
from utilsProject import convert_from_image_to_cv2
import cv2
# import torch


def main(image, segment=False, save_extracted_object = False):

    """

    :param image:
    :param segment:
    :param save_extracted_object:
    :return: returns a json object which contains analyzed data of an image

    note : we will have to update this method based on how we connect to the other part of the application
    for now just by running this we can get the results
    more work to be done in future
    """
    # image = convert_json_to_image(image)

    global input_image
    if segment:
        image = image.convert("RGB")
        segmented_image = seperate_objs_from_image(image,save_extracted_object)
        input_image = segmented_image
    elif segment== False:
        input_image = convert_from_image_to_cv2(image)

    input_image = pre_process(input_image, segment)

    clusters = meanshift(input_image)

    # sillhouttle_analysis(input_image,clusters[0],clusters[1],len(clusters[2]))
    #
    json_object = post_processing(image,clusters[0], clusters[1], segment)

    print(json_object)


if __name__ == '__main__':

    """
    Testing implementation
    """
    input_image = Image.open('samples/sample3.jpg')
    print(input_image.size)
    t1 = time.perf_counter()
    main(input_image)
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

