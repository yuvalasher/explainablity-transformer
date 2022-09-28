from pathlib import Path

import torch
from torch.functional import F
from PIL import Image
from pytorch_lightning import seed_everything
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
seed_everything(42)

model_name = "google/vit-base-patch16-224"
IMAGES_FOLDER_PATH = ""

model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

l = os.listdir(IMAGES_FOLDER_PATH)
for image_name in l:
    image = Image.open(Path(IMAGES_FOLDER_PATH, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    correct_class_prob = F.softmax(logits[0], dim=-1)[predicted_class_idx].item()
    correct_class_logits = torch.max(logits[0])
    print(
        image_name.split('.')[0],
        model.config.id2label[predicted_class_idx],
        correct_class_prob,
        correct_class_logits.item(),
    )
    probs, indices = torch.topk(F.softmax(logits[0], dim=-1), k=10, largest=True)
    for prob, ind in zip(probs, indices):
        print(
            f"Class: {model.config.id2label[ind.item()]}, Prob: {prob.item()}, Class Idx: {ind.item()}"
        )
    print(
        "----------------------------------------------------------------------------------------------------------------"
    )
