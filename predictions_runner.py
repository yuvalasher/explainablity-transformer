from utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything

vit_config = config['vit']
seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])

l = ['chicken.png', '0000000005.JPEG', '0000000002.JPEG', '00000112.JPEG','00000106.JPEG', '0000000003.JPEG', '0000000004.JPEG','0000000002.JPEG', '0000000001.JPEG','0000000005.JPEG', '00000082.JPEG', '00000032.JPEG', '00000052.JPEG', '00000069.JPEG' , '00000009.JPEG', '00000009.JPEG', '00000018.JPEG', '00000327.JPEG']

for image_name in l:
    image = get_image_from_path(os.path.join(IMAGES_FOLDER_PATH, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = vit_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    correct_class_prob = F.softmax(logits[0], dim=-1)[predicted_class_idx].item()
    correct_class_logits = torch.max(logits[0])
    print(image_name, vit_model.config.id2label[predicted_class_idx], correct_class_prob, correct_class_logits.item())
    probs, indices = torch.topk(F.softmax(logits[0], dim=-1), k=10, largest=True)
    for prob, ind in zip(probs, indices):
        print(f'Class: {vit_model.config.id2label[ind.item()]}, Prob: {prob.item()}, Class Idx: {ind.item()}')
    print(
        '----------------------------------------------------------------------------------------------------------------')
