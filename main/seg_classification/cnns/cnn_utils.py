from torchvision import models
from PIL import Image
from torchvision import transforms

RESNET_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
RESNET_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def resize_center_crop(image):
    return resnet_resize_center_crop_transform(image)


def resize_center_crop_normalize(image):
    return resnet_preprocess(image)


resnet_resize_center_crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), ])

resnet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=RESNET_NORMALIZATION_MEAN,
        std=RESNET_NORMALIZATION_STD
    )])
