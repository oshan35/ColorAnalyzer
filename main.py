from clustering import meanshift
from predict import segmentation
from utils import pre_process
from utils import post_processing
from PIL import Image
from utils import convert_from_image_to_cv2
import time
from glob import glob
from tqdm import tqdm
import cv2


def color_analysis_pipeline(image, name, segment_humans=False, save_extracted_object=False):
    global input_image
    # image = convert_from_image_to_cv2(image)
    input_image = image
    if segment_humans:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmented_image = segmentation(image, name, save_extracted_object)
        input_image = segmented_image

    input_image = pre_process(input_image, segment_humans)

    clusters = meanshift(input_image)

    json_object = post_processing(image, clusters[0], clusters[1])

    return json_object


def main():
    t1 = time.perf_counter()

    sample_dict = glob('sample-02/*')

    return_dict = {}

    for path in tqdm(sample_dict, total=len(sample_dict)):
        name = path.split("\\")[-1].split(".")[0]
        input_image = cv2.imread(path, cv2.IMREAD_COLOR)

        json_object = color_analysis_pipeline(input_image, name, segment_humans=True, save_extracted_object=True)

        return_dict[name] = json_object
    print(return_dict)
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')


if __name__ == '__main__':
    main()
