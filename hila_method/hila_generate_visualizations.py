from typing import List, Dict

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.perturbation_tests.seg_cls_perturbation_tests import run_perturbation_test
from hila_method.utils.ViT_LRP import vit_base_patch16_224 as vit_LRP
from hila_method.utils.ViT_explanation_generator import LRP
from hila_method.utils.imagenet_dataset import ImageNetDataset
import torch

from vit_loader.load_vit import load_vit_pretrained
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
from torchvision.transforms import transforms


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(dataloader: DataLoader) -> List[Dict[str, Tensor]]:
    outputs: List[Dict[str, Tensor]] = []
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        # target = target.to(device)
        data = normalize(data)
        data = data.to(device)
        data.requires_grad_()
        Res = lrp.generate_LRP(data, start_layer=1, method="grad", index=None).reshape(data.shape[0], 1, 14, 14)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
        Res = (Res - Res.min()) / (Res.max() - Res.min())  # TODO - check if should do it on ours!
        # Res_np = Res.data.cpu().numpy()
        data = data.cpu()
        Res = Res.data.cpu()
        outputs.append({'original_image': data, 'image_mask': Res})
    return outputs


if __name__ == '__main__':
    IMAGENET_VALIDATION_PATH = "/home/yuvalas/explainability/data/run_hila_3000"
    BATCH_SIZE = 1
    MODEL_NAME = 'google/vit-base-patch16-224'
    model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    vit_for_classification_image, _ = load_vit_pretrained(model_name=MODEL_NAME)

    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, transform=transform)
    # print(imagenet_ds[0])
    # imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    outputs = compute_saliency_and_save(dataloader=sample_loader)
    auc = run_perturbation_test(
        model=vit_for_classification_image,
        outputs=outputs,
        stage="hila",
        epoch_idx=0,
    )
