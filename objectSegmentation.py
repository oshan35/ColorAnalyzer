import torch
from torchvision import transforms
import cv2
import numpy as np
import matplotlib as plt
from PIL import Image


def load_model():
    """
        This method loads the DeeplabV3 model from pytorch libs
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.to(device).eval()
    return model


def get_predictions(image, model):
    # If GPU is available and if use it else go with cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output = output.argmax(0)
    return output.cpu().numpy()


def seperate_objs_from_image(image, save_extracted_object=False):

    # reduce the image size by half. This is necessary for big images more then 10MB
    if image.size[0] > 2000 or image.size[1] > 2000:
        x,y = int(image.size[0]/1000), int(image.size[1]/1000)
        reduce_factor = max(x,y)
        image = image.resize((int(image.size[0]/reduce_factor),int(image.size[1]/reduce_factor)), Image.ANTIALIAS)

    deeplabv3 = load_model()

    labels = get_predictions(image, deeplabv3)

    mask = labels == 15

    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    segmented_image = image * mask
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    if save_extracted_object:
        cv2.imwrite('extracted_objects/sample02.jpg', segmented_image)
    return segmented_image
