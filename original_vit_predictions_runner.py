from torch import nn
from torch.functional import F
from torch import optim
from config import config
from utils import *
from vit_utils import *
from loss_utils import *
from log_utils import configure_log
from consts import *
from pytorch_lightning import seed_everything

vit_config = config['vit']
seed_everything(config['general']['seed'])
feature_extractor, vit_model, vit_sigmoid_model = load_feature_extractor_and_vit_model(vit_config=vit_config)

for idx, image_name in enumerate(os.listdir(images_folder_path)):
    image = get_image_from_path(os.path.join(images_folder_path, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = vit_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print(image_name, vit_model.config.id2label[predicted_class_idx], F.softmax(logits[0])[predicted_class_idx].item())
    if idx == 20:
        break