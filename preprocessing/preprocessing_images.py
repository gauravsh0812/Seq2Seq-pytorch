import torch, os
from torchvision import transforms
from PIL import Image

def crop_image(image):
    return transforms.functional.crop(image, 0, 0, width, height)


def preprocess(img_batch, datapath):
    """
    Crop, padding, and downsample the image.
    :params img_batch: batch of images
    :return: processed images
    """
    for image_label in img_batch:
        # opening the image
        IMAGE = Image(os.path.join(datapath, image_label))

        # build pipeline to trsndorm the image
        data_transforms = {
            'image': transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(crop_image),
                              transforms.Pad())])}
