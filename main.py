import time

from clustering import meanshift
import matplotlib.pyplot as plt
from utilsProject import pre_process
from utilsProject import post_processing
from PIL import Image
from utilsProject import convert_json_to_image
from utilsProject import convert_from_image_to_cv2
import cv2
# import torch
from predict import segmentation
from glob import glob
from tqdm import tqdm

def main(images, segment=False, save_extracted_object=False):
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
        # image = image.convert("RGB")
        segmented_image = segmentation(images,save_extracted_object)
        input_image = segmented_image
    elif segment == False:
        for name, image in images.items():
            img = convert_from_image_to_cv2(image)
            input_image[name] = img
    # for name, image in images.items():
    #     plt.imshow(image)
    #     plt.show()

    cluster_results = {}
    for name, image in tqdm(input_image.items(), total=len(images)):
        input_image[name] = pre_process(image, segment)

        clusters = meanshift(input_image[name])
        cluster_results[name] = clusters


    # sillhouttle_analysis(input_image,clusters[0],clusters[1],len(clusters[2]))
    #

    for name, cluster in cluster_results.items():

        json_object = post_processing(images[name], cluster[0], cluster[1])


        print(json_object)


if __name__ == '__main__':

    """
    Testing implementation
    """

    t1 = time.perf_counter()
    inpux_x = glob('samples/*')
    in_images = {}
    for path in inpux_x:
        name = path.split("\\")[-1].split(".")[0]

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        in_images[name] = image
    main(in_images,segment=True, save_extracted_object=True)
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')
