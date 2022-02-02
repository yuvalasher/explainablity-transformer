from modeling_vit_sigmoid import ViTSigmoidForImageClassification
from transformers import ViTConfig
import torch
from torch import nn
from torch.functional import F
from config import config
from utils import *
from vit_utils import *
from loss_utils import *
from consts import *
from pytorch_lightning import seed_everything
from scipy.stats import bernoulli

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])


def print_conclusions(vit_model, tokens_mask, output, target) -> None:
    print(
        f'Num of patches: {tokens_mask.sum()}, {round((tokens_mask.sum() / len(tokens_mask)).item(), 2) * 100}% of the tokens, '
        f'correct_class_pred: {F.softmax(output.logits)[0][torch.argmax(F.softmax(target.logits)).item()]}, '
        f'correct_class_logit: {output.logits[0][torch.argmax(F.softmax(target.logits[0])).item()]}, '
        f'Highest class: {torch.argmax(output.logits[0])} , {vit_model.config.id2label[torch.argmax(output.logits[0]).item()]} with {torch.max(F.softmax(output.logits[0])).item()}, '
        f'Is highest class: {torch.argmax(output.logits[0]) == torch.argmax(target.logits[0])}')


def load_obj(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model: nn.Module, path: str) -> None:
    path = Path(f'{path}.pt')
    torch.save(model.state_dict(), path)
    print(f'Model Saved at {path}')


def load_model(path: str) -> nn.Module:
    # path = Path(f'{PICKLES_FOLDER_PATH}', f'{model_name}.pt')
    if path[-3:] == '.pt':
        path = Path(f'{path}')
    else:
        path = Path(f'{path}.pt')
    c = ViTConfig()
    c.image_size = vit_config['img_size']
    c.num_labels = vit_config['num_labels']
    model = ViTSigmoidForImageClassification(config=c)
    model.load_state_dict(torch.load(path))
    return model


def dark_random_k_patches(precentage_to_dark: float, n_patches: int = 577) -> Tensor:
    k = int(n_patches * precentage_to_dark)
    random_vector = torch.rand(n_patches)
    k_th_quant = torch.topk(random_vector, k, largest=False)[0][-1]
    mask = (random_vector >= k_th_quant).int()
    return mask


def _load_extract_features(model_path: str):
    feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
    vit_sigmoid_model = load_model(path=model_path)
    image = get_image_from_path(os.path.join(images_folder_path, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    target = vit_model(**inputs)
    return inputs, target, vit_model, vit_sigmoid_model


def test_dark_random_k_patches(path, num_tests, precentage_to_dark: float):
    correct_random_guess = []
    inputs, target, vit_model, vit_sigmoid_model = _load_extract_features(model_path=path)
    for test_idx in range(num_tests):
        tokens_mask = dark_random_k_patches(precentage_to_dark=precentage_to_dark)
        output = vit_sigmoid_model(**inputs, tokens_mask=tokens_mask)
        if torch.argmax(output.logits[0]) == torch.argmax(target.logits[0]):
            correct_random_guess.append((test_idx, torch.max(F.softmax(output.logits[0])).item()))
        print(f'*** {test_idx} ***')
        print_conclusions(vit_model, tokens_mask, output, target)
    print(f'Number of correct random guesses: {len(correct_random_guess)}')
    print(correct_random_guess)


def infer():
    feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
    vit_s = ViTSigmoidForImageClassification.from_pretrained(vit_config['model_name'])
    image = get_image_from_path(os.path.join(images_folder_path, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    target = vit_model(**inputs)
    output = vit_s(**inputs, tokens_mask=torch.ones(577))
    print(
        f'correct_class_pred: {F.softmax(output.logits)[0][torch.argmax(F.softmax(target.logits)).item()]}, correct_class_logit: {output.logits[0][torch.argmax(F.softmax(target.logits[0])).item()]}')


def generate_sampled_binary_patches_from_distribution(distribution: Tensor, precentage_to_dark: float) -> Tensor:
    dist = distribution.clone()
    return dist.detach().apply_(lambda x: bernoulli.rvs(x, size=1)[0])
    # return dist.detach().apply_(lambda x: 1 if x > 0 else 0)
    # return torch.tensor([bernoulli.rvs(p.item(),size=1)[0] for p in dist])


def generate_sampled_binary_patches_by_top_scores(distribution: Tensor, tokens_to_show: int) -> Tensor:
    dist = distribution.clone()
    k = tokens_to_show  # int(len(dist) * precentage_to_dark)
    # k_th_quant = torch.topk(dist, k, largest=False)[0][-1]
    # mask = (dist <= k_th_quant).int()
    k_th_quant = torch.topk(dist, k)[0][-1]
    mask = (dist >= k_th_quant).int()
    return mask


def get_dino_probability_per_head(path: str, attentions_path: str, tokens_to_show: int):
    attentions = load_obj(path=attentions_path)
    inputs, target, vit_model, vit_sigmoid_model = _load_extract_features(model_path=path)

    for attention_head in attentions:
        tokens_mask = generate_sampled_binary_patches_by_top_scores(distribution=attention_head,
                                                                    tokens_to_show=tokens_to_show)
        tokens_mask = torch.cat((torch.ones(1), tokens_mask))  # one for the [CLS] token
        output = vit_sigmoid_model(**inputs, tokens_mask=tokens_mask)
        print_conclusions(vit_model, tokens_mask, output, target)


def infer_prediction(path: str, tokens_mask: Tensor = None, experiment_name: str = None):
    inputs, target, vit_model, vit_sigmoid_model = _load_extract_features(model_path=path)

    if tokens_mask is not None:
        output = vit_sigmoid_model(**inputs, tokens_mask=tokens_mask)
    else:
        output = vit_sigmoid_model(**inputs)
    print_conclusions(vit_model, tokens_mask, output, target)
    print(1)


snake_tokens_mask_obj_1 = torch.tensor([0.7155, 0.5070, 0.3448, 0.3422, 0.3352, 0.3436, 0.3426, 0.3434, 0.3129,
                                        0.3835, 0.3456, 0.3809, 0.3812, 0.3777, 0.2848, 0.3387, 0.3512, 0.3117,
                                        0.3366, 0.3064, 0.3377, 0.3471, 0.3402, 0.3448, 0.3386, 0.3479, 0.3761,
                                        0.5118, 0.4144, 0.4164, 0.3090, 0.4144, 0.3487, 0.3133, 0.3745, 0.4142,
                                        0.4102, 0.3099, 0.4068, 0.4168, 0.3744, 0.3707, 0.3166, 0.3752, 0.3152,
                                        0.3135, 0.3386, 0.4206, 0.4116, 0.3438, 0.2864, 0.3667, 0.3117, 0.3512,
                                        0.3483, 0.3759, 0.3434, 0.3891, 0.4160, 0.4131, 0.3814, 0.4102, 0.4503,
                                        0.3361, 0.3404, 0.3694, 0.4177, 0.4116, 0.4102, 0.3800, 0.3749, 0.4218,
                                        0.4598, 0.4589, 0.2664, 0.3178, 0.4119, 0.3459, 0.3461, 0.3118, 0.4122,
                                        0.3158, 0.4086, 0.4227, 0.4154, 0.4575, 0.3723, 0.3407, 0.4067, 0.4198,
                                        0.4164, 0.3340, 0.3776, 0.4181, 0.3189, 0.3701, 0.3841, 0.4485, 0.2975,
                                        0.3718, 0.2705, 0.3723, 0.3450, 0.3806, 0.3724, 0.3683, 0.3715, 0.4123,
                                        0.4643, 0.4593, 0.4645, 0.4966, 0.3354, 0.3128, 0.4140, 0.4088, 0.4142,
                                        0.3686, 0.4207, 0.4076, 0.4980, 0.4112, 0.3051, 0.4893, 0.4061, 0.3354,
                                        0.4085, 0.4092, 0.4559, 0.4504, 0.4185, 0.2944, 0.2542, 0.2679, 0.2905,
                                        0.4518, 0.3080, 0.4145, 0.3352, 0.3813, 0.5018, 0.2353, 0.3347, 0.5073,
                                        0.4947, 0.3772, 0.2946, 0.3754, 0.2784, 0.3781, 0.3177, 0.5121, 0.4533,
                                        0.2179, 0.5043, 0.3788, 0.2793, 0.4121, 0.4589, 0.2345, 0.2697, 0.4618,
                                        0.4183, 0.2632, 0.5001, 0.4942, 0.4093, 0.4082, 0.3102, 0.4182, 0.4110,
                                        0.2145, 0.2653, 0.3696, 0.4517, 0.4546, 0.3424, 0.9490, 0.6381, 0.2054,
                                        0.5063, 0.3176, 0.4417, 0.4466, 0.4543, 0.4125, 0.9578, 0.3917, 0.1490,
                                        0.9944, 0.1826, 0.3404, 0.3169, 0.4058, 0.4714, 0.4191, 0.5040, 0.3835,
                                        0.2488, 0.4480, 0.3770, 0.4129, 0.1968, 0.9608, 0.3060, 0.9943, 0.9877,
                                        0.2361, 0.8001, 0.1731, 0.9855, 0.2469, 0.4528, 0.3791, 0.3442, 0.3755,
                                        0.3448, 0.4541, 0.3780, 0.3066, 0.3446, 0.4594, 0.3230, 0.4422, 0.1817,
                                        0.8226, 0.3453, 0.6827, 0.6851, 0.4529, 0.4275, 0.2330, 0.2978, 0.7744,
                                        0.2275, 0.3135, 0.4977, 0.4586, 0.3438, 0.3351, 0.4145, 0.4568, 0.4138,
                                        0.4958, 0.4564, 0.3055, 0.9556, 0.1037, 0.8843, 0.8580, 0.8935, 0.7096,
                                        0.9964, 0.4719, 0.5089, 0.5328, 0.7966, 0.9106, 0.5237, 0.5088, 0.4514,
                                        0.4067, 0.4187, 0.3384, 0.3806, 0.3781, 0.4116, 0.3515, 0.5965, 0.4723,
                                        0.9874, 0.0661, 0.9922, 0.6343, 0.1984, 0.4322, 0.6662, 0.2671, 0.3070,
                                        0.2124, 0.2551, 0.3029, 0.4545, 0.3719, 0.3826, 0.3871, 0.3744, 0.3479,
                                        0.2961, 0.3404, 0.4132, 0.5060, 0.4089, 0.3750, 0.2999, 0.1748, 0.9995,
                                        0.4114, 0.3489, 0.4597, 0.3482, 0.2617, 0.3414, 0.3715, 0.4084, 0.3434,
                                        0.3887, 0.3389, 0.3765, 0.3762, 0.2774, 0.3081, 0.2847, 0.4126, 0.3858,
                                        0.3676, 0.3804, 0.5064, 0.3001, 0.2924, 0.3531, 0.3811, 0.2769, 0.3445,
                                        0.4123, 0.3142, 0.3826, 0.3380, 0.3760, 0.3483, 0.3141, 0.3409, 0.4636,
                                        0.3494, 0.3779, 0.4386, 0.4264, 0.4194, 0.4069, 0.5010, 0.4126, 0.3463,
                                        0.3408, 0.3428, 0.5282, 0.4218, 0.4211, 0.3708, 0.3109, 0.3778, 0.4160,
                                        0.3477, 0.3847, 0.3443, 0.4155, 0.3475, 0.3439, 0.4181, 0.3371, 0.3461,
                                        0.3774, 0.4291, 0.4557, 0.3090, 0.4056, 0.3389, 0.3474, 0.4121, 0.2634,
                                        0.3193, 0.3510, 0.3078, 0.3675, 0.3818, 0.3725, 0.3475, 0.3758, 0.3725,
                                        0.4046, 0.3716, 0.3188, 0.4626, 0.3765, 0.4218, 0.3094, 0.4608, 0.4055,
                                        0.4974, 0.3174, 0.4114, 0.3386, 0.4564, 0.4170, 0.5078, 0.2935, 0.3800,
                                        0.3344, 0.3746, 0.3808, 0.2635, 0.3247, 0.2601, 0.4602, 0.3339, 0.3711,
                                        0.4102, 0.3814, 0.3733, 0.4297, 0.3690, 0.2944, 0.3740, 0.3778, 0.3749,
                                        0.2941, 0.3091, 0.4584, 0.4268, 0.3685, 0.4166, 0.3690, 0.3833, 0.2369,
                                        0.4518, 0.3004, 0.2227, 0.2780, 0.4190, 0.4191, 0.4115, 0.4059, 0.4186,
                                        0.3481, 0.3193, 0.4124, 0.2616, 0.3412, 0.4629, 0.4632, 0.3079, 0.3664,
                                        0.3475, 0.4234, 0.3075, 0.4086, 0.3427, 0.4195, 0.3760, 0.3724, 0.1958,
                                        0.2194, 0.3657, 0.3780, 0.3789, 0.4574, 0.3478, 0.4063, 0.4158, 0.4635,
                                        0.5022, 0.2960, 0.3675, 0.3865, 0.2667, 0.3427, 0.3143, 0.5026, 0.4599,
                                        0.3857, 0.4222, 0.3715, 0.3415, 0.4088, 0.4627, 0.3679, 0.3842, 0.4604,
                                        0.4119, 0.3734, 0.3802, 0.3752, 0.3853, 0.4143, 0.4049, 0.3811, 0.3415,
                                        0.4585, 0.3686, 0.5079, 0.3266, 0.3820, 0.4091, 0.3503, 0.3737, 0.3742,
                                        0.4621, 0.4160, 0.3784, 0.4112, 0.3737, 0.4169, 0.3702, 0.3686, 0.3462,
                                        0.3311, 0.3401, 0.3798, 0.3839, 0.3705, 0.3455, 0.3691, 0.3356, 0.3777,
                                        0.3778, 0.4592, 0.4139, 0.3538, 0.3824, 0.4069, 0.3803, 0.3815, 0.4600,
                                        0.3768, 0.4622, 0.4073, 0.4115, 0.3771, 0.3764, 0.3826, 0.3357, 0.4170,
                                        0.3255, 0.3723, 0.3723, 0.3847, 0.4167, 0.3404, 0.3799, 0.4403, 0.4648,
                                        0.3812, 0.3183, 0.4241, 0.3782, 0.3466, 0.3691, 0.3708, 0.3798, 0.3770,
                                        0.3378, 0.3450, 0.5086, 0.3490, 0.2525, 0.3797, 0.4197, 0.4120, 0.4155,
                                        0.3779, 0.4060, 0.3751, 0.3772, 0.3840, 0.3370, 0.3509, 0.4180, 0.3757,
                                        0.4228, 0.3797, 0.4127, 0.3147, 0.4161, 0.3485, 0.4089, 0.5092, 0.5079,
                                        0.3189])
snake_tokens_mask_obj_2 = torch.tensor([0.9133, 0.8870, 0.9319, 0.9386, 0.9177, 0.9129, 0.9109, 0.9301, 0.8819,
                                        0.3064, 0.3048, 0.8994, 0.9469, 0.8815, 0.9180, 0.2233, 0.9520, 0.8392,
                                        0.9066, 0.8530, 0.4667, 0.2910, 0.9214, 0.3452, 0.9435, 0.9368, 0.9186,
                                        0.2538, 0.3883, 0.1835, 0.1727, 0.9003, 0.9441, 0.9323, 0.8941, 0.9259,
                                        0.7815, 0.8755, 0.4114, 0.1833, 0.8998, 0.8921, 0.9238, 0.9364, 0.8842,
                                        0.8533, 0.9422, 0.9437, 0.9058, 0.9400, 0.9117, 0.8094, 0.8798, 0.4988,
                                        0.9056, 0.9265, 0.7845, 0.8980, 0.2896, 0.1555, 0.8920, 0.8053, 0.7520,
                                        0.9082, 0.9164, 0.8330, 0.5236, 0.8710, 0.2878, 0.3114, 0.9155, 0.8477,
                                        0.2385, 0.9399, 0.1961, 0.7737, 0.4993, 0.8059, 0.8898, 0.9050, 0.4418,
                                        0.9359, 0.9331, 0.4745, 0.9289, 0.4314, 0.3344, 0.1886, 0.9346, 0.2800,
                                        0.8946, 0.9086, 0.4656, 0.8853, 0.4288, 0.8391, 0.9504, 0.9462, 0.8504,
                                        0.9462, 0.9459, 0.2556, 0.9144, 0.9450, 0.8174, 0.3683, 0.8725, 0.4935,
                                        0.7766, 0.9375, 0.1986, 0.9165, 0.9461, 0.2088, 0.2426, 0.7810, 0.7835,
                                        0.4026, 0.9174, 0.1927, 0.8606, 0.9488, 0.9050, 0.3097, 0.9163, 0.8339,
                                        0.9461, 0.1788, 0.9298, 0.9415, 0.8984, 0.2149, 0.1665, 0.8668, 0.3947,
                                        0.9311, 0.2497, 0.2168, 0.3686, 0.9416, 0.9296, 0.5440, 0.9393, 0.9224,
                                        0.9141, 0.9019, 0.9417, 0.2280, 0.9071, 0.9364, 0.9337, 0.8407, 0.9320,
                                        0.2027, 0.2583, 0.8997, 0.9083, 0.9401, 0.9074, 0.8771, 0.1154, 0.9264,
                                        0.8933, 0.9200, 0.1668, 0.1711, 0.9180, 0.8959, 0.7953, 0.9201, 0.9193,
                                        0.8281, 0.1940, 0.9356, 0.2120, 0.9051, 0.1316, 0.9082, 0.1943, 0.2906,
                                        0.1434, 0.8951, 0.9096, 0.2122, 0.1599, 0.5872, 0.9172, 0.5478, 0.1181,
                                        0.3498, 0.9296, 0.9408, 0.9484, 0.9288, 0.8035, 0.9393, 0.8980, 0.9115,
                                        0.8860, 0.2165, 0.1908, 0.1737, 0.9050, 0.1642, 0.9087, 0.1673, 0.9115,
                                        0.9014, 0.1588, 0.2033, 0.1995, 0.2063, 0.7376, 0.1658, 0.8440, 0.8926,
                                        0.9427, 0.8833, 0.9161, 0.3495, 0.9499, 0.9304, 0.9113, 0.1780, 0.1582,
                                        0.2143, 0.2575, 0.1714, 0.8824, 0.1544, 0.1536, 0.1504, 0.1624, 0.2656,
                                        0.1515, 0.2120, 0.9151, 0.2719, 0.2594, 0.8453, 0.4771, 0.9096, 0.1533,
                                        0.9512, 0.2041, 0.9340, 0.3271, 0.1693, 0.1414, 0.1627, 0.8925, 0.3653,
                                        0.1902, 0.1964, 0.9016, 0.7973, 0.1650, 0.1949, 0.9145, 0.8368, 0.9387,
                                        0.8991, 0.9438, 0.8552, 0.9231, 0.8560, 0.8520, 0.8723, 0.9315, 0.9221,
                                        0.1764, 0.1857, 0.0974, 0.1670, 0.9366, 0.9200, 0.7521, 0.1969, 0.2286,
                                        0.8492, 0.8500, 0.8146, 0.9079, 0.9464, 0.9321, 0.8752, 0.9481, 0.9262,
                                        0.8683, 0.8837, 0.9420, 0.9144, 0.8998, 0.9349, 0.4550, 0.1314, 0.3042,
                                        0.9390, 0.9252, 0.4273, 0.8210, 0.8912, 0.9061, 0.1797, 0.9279, 0.8077,
                                        0.1542, 0.9502, 0.9320, 0.9301, 0.8489, 0.9345, 0.8778, 0.9076, 0.9136,
                                        0.9066, 0.3875, 0.9367, 0.4833, 0.8288, 0.9293, 0.2624, 0.8920, 0.2307,
                                        0.4939, 0.9215, 0.3142, 0.9175, 0.9102, 0.9378, 0.9026, 0.3740, 0.7907,
                                        0.9275, 0.9516, 0.9543, 0.9491, 0.8384, 0.8866, 0.8345, 0.9510, 0.1871,
                                        0.9181, 0.8747, 0.8021, 0.8959, 0.1859, 0.9219, 0.8885, 0.9236, 0.9290,
                                        0.4039, 0.3077, 0.9362, 0.3669, 0.8762, 0.2763, 0.8816, 0.7610, 0.9206,
                                        0.7653, 0.9425, 0.8982, 0.9339, 0.5850, 0.9086, 0.3272, 0.7205, 0.8333,
                                        0.8579, 0.2947, 0.8552, 0.7998, 0.9111, 0.7883, 0.2786, 0.7632, 0.8786,
                                        0.8986, 0.8382, 0.9036, 0.9031, 0.7799, 0.9240, 0.9368, 0.7782, 0.9342,
                                        0.2438, 0.9205, 0.2418, 0.9139, 0.9563, 0.8475, 0.9171, 0.2900, 0.8457,
                                        0.7359, 0.9196, 0.1526, 0.9251, 0.1768, 0.9287, 0.9297, 0.8755, 0.8447,
                                        0.9024, 0.7756, 0.9366, 0.9267, 0.5256, 0.2528, 0.7490, 0.9327, 0.4163,
                                        0.9428, 0.3383, 0.8740, 0.2220, 0.9216, 0.7923, 0.9389, 0.2187, 0.9171,
                                        0.2104, 0.9028, 0.9167, 0.9442, 0.9307, 0.9328, 0.8964, 0.8266, 0.9145,
                                        0.8786, 0.9335, 0.8780, 0.9279, 0.3981, 0.9275, 0.8355, 0.1779, 0.9394,
                                        0.9173, 0.3563, 0.8359, 0.8487, 0.9399, 0.4501, 0.9289, 0.8220, 0.1881,
                                        0.3084, 0.8084, 0.9187, 0.9218, 0.9387, 0.9182, 0.9438, 0.8148, 0.8552,
                                        0.8658, 0.5083, 0.8550, 0.7479, 0.8612, 0.4794, 0.8845, 0.1614, 0.7490,
                                        0.2222, 0.1850, 0.9043, 0.8421, 0.8791, 0.2222, 0.9283, 0.8822, 0.8709,
                                        0.9519, 0.8507, 0.4437, 0.5148, 0.2376, 0.8999, 0.8346, 0.2834, 0.2962,
                                        0.2483, 0.3723, 0.8662, 0.8799, 0.9220, 0.3626, 0.4203, 0.2581, 0.9266,
                                        0.4951, 0.3789, 0.9214, 0.1457, 0.9304, 0.8470, 0.8874, 0.8242, 0.4585,
                                        0.9136, 0.5366, 0.3422, 0.4817, 0.2354, 0.5023, 0.2128, 0.8280, 0.2149,
                                        0.4162, 0.9186, 0.8730, 0.2071, 0.2756, 0.5026, 0.8900, 0.9246, 0.4198,
                                        0.2540, 0.9120, 0.8843, 0.8823, 0.9235, 0.2994, 0.9134, 0.8779, 0.9175,
                                        0.8201, 0.9241, 0.8695, 0.9330, 0.9285, 0.4064, 0.3236, 0.8864, 0.3248,
                                        0.9043, 0.1908, 0.8963, 0.2680, 0.4369, 0.9240, 0.2195, 0.9403, 0.8815,
                                        0.2112, 0.9412, 0.2671, 0.3423, 0.9219, 0.8979, 0.8710, 0.3741, 0.4886,
                                        0.8981, 0.9109, 0.8188, 0.8006, 0.9050, 0.8960, 0.9337, 0.2773, 0.4696,
                                        0.8133, 0.7853, 0.8761, 0.9386, 0.4896, 0.8561, 0.8779, 0.2493, 0.9602,
                                        0.9554])
snake_tokens_mask_opposite_gumble = torch.tensor(
    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0.,
     0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.,
     1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.,
     1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 0., 0.,
     0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1.,
     1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
     0.])
# snake_tokens_mask_gumble = torch.tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
#         0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
#         1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
#         1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
#         0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0.,
#         0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
#         1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.,
#         1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
#         0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0.,
#         0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
#         0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
#         1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0.,
#         1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1.,
#         0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.,
#         0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1.,
#         1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0.,
#         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
#         0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
#         0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0.,
#         0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
#         1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
#         0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
#         0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
#         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
#         0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
#         1.])
snake_tokens_mask_gumble = torch.tensor([1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0.,
                                         0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.,
                                         0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                         0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
                                         0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1.,
                                         0., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0.,
                                         0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
                                         1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
                                         0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
                                         0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1.,
                                         0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1.,
                                         1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0.,
                                         1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1.,
                                         1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0.,
                                         0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1.,
                                         1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
                                         0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
                                         0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                         0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0.,
                                         0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                                         0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
                                         1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1.,
                                         0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
                                         0.])
bird_tokens_mask_gumble = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
                                        1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
                                        0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.,
                                        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                                        1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
                                        1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
                                        1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                        1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0.,
                                        1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0.,
                                        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                        0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                        0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.,
                                        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1.,
                                        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
                                        0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
                                        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                        0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
                                        0.])
dog_tokens_mask_gumble = torch.tensor([1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0.,
                                       1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1.,
                                       1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0.,
                                       0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                       0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
                                       0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
                                       0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
                                       0., 1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0.,
                                       0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1.,
                                       1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1.,
                                       1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1.,
                                       1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1.,
                                       0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1.,
                                       0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0., 0.,
                                       0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0.,
                                       0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0.,
                                       0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1.,
                                       0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
                                       0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 0.,
                                       1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 0., 1.,
                                       0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1.,
                                       1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 1.,
                                       1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
                                       0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1.,
                                       1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.,
                                       1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1.,
                                       0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                       1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
                                       1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1.,
                                       0.])
dino_snake_attentions_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\dino method on supervised vit - cls of the last layer to other tokens\ILSVRC2012_val_00000001\original\attentions.pkl"
dino_bird_attentions_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\dino method on supervised vit - cls of the last layer to other tokens\ILSVRC2012_val_00000018\original\dino_bird_attentions.pkl"
dino_dog_attentions_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\dino method on supervised vit - cls of the last layer to other tokens\ILSVRC2012_val_00000003\attentions.pkl"
model_obj_1_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_1_lr0_3_temp_1+l1_1+kl_loss_0+entropy_loss_5+pred_loss_10\ILSVRC2012_val_00000001\vit_sigmoid_model.pt"
model_obj_2_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_2_lr0_3_temp_1+l1_1+kl_loss_0+entropy_loss_3+pred_loss_3\ILSVRC2012_val_00000001\vit_sigmoid_model.pt"
model_gumble_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_0.001+entropy_loss_0+pred_loss_3\ILSVRC2012_val_00000018\vit_sigmoid_model.pt"
model_gumble_opposite_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\ILSVRC2012_val_00000001\vit_sigmoid_model.pt"
# Model Saved at C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\ILSVRC2012_val_00000001\vit_sigmoid_model.pt

if __name__ == '__main__':
    print('snake')
    infer_prediction(path=model_gumble_path, tokens_mask=torch.ones_like(snake_tokens_mask_gumble))
    infer_prediction(path=model_gumble_path, tokens_mask=snake_tokens_mask_gumble)
    # test_dark_random_k_patches(path=model_gumble_path, num_tests=1000, precentage_to_dark=0.81)
    get_dino_probability_per_head(path=model_gumble_path, attentions_path=dino_snake_attentions_path,
                                  tokens_to_show=int(snake_tokens_mask_gumble.sum().item()))

    # print('dog')
    # infer_prediction(path=model_gumble_path, tokens_mask=torch.ones_like(dog_tokens_mask_gumble))
    # infer_prediction(path=model_gumble_path, tokens_mask=dog_tokens_mask_gumble)
    # # test_dark_random_k_patches(path=model_gumble_path, num_tests=1000, precentage_to_dark=0.81)
    # get_dino_probability_per_head(path=model_gumble_path, attentions_path=dino_dog_attentions_path,
    #                               tokens_to_show=int(dog_tokens_mask_gumble.sum().item()))
