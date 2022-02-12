from utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything

vit_config = config['vit']
seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)

for idx, image_name in enumerate(os.listdir(IMAGES_FOLDER_PATH)[30:]):
    try:
        image = get_image_from_path(os.path.join(IMAGES_FOLDER_PATH, image_name))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = vit_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        correct_class_prob = F.softmax(logits[0])[predicted_class_idx].item()
        correct_class_logits = torch.max(logits[0])
        if correct_class_prob <= 0.85 and correct_class_prob >= 0.75:
            print(image_name, vit_model.config.id2label[predicted_class_idx], correct_class_prob, correct_class_logits)
    except:
        pass
    if idx == 700:
        break
