from torchvision import transforms
from config import config

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

# normalize = transforms.Normalize(mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD)
#
# image_transformations = transforms.Compose([
#     transforms.PILToTensor(),
#     transforms.Resize((config['vit']['img_size'], config['vit']['img_size'])),
#     normalize,
# ])

normalize = transforms.Normalize(mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD)
image_transformations = transforms.Compose([
    transforms.Resize((config['vit']['img_size'], config['vit']['img_size'])),
    transforms.ToTensor(),
    normalize,
])