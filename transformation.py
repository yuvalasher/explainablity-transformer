from torchvision import transforms
from config import config

pil_to_resized_tensor_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((config['vit']['img_size'], config['vit']['img_size']))
])
