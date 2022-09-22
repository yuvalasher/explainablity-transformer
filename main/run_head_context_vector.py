import pandas as pd
from torchvision.transforms import transforms
from tqdm import tqdm

# from utils.utils import *
from evaluation.evaluation_utils import load_obj_from_path
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_temp_softmax, objective_grad_rollout
from evaluation.perturbation_tests.head_perturbation_tests import (
    eval,
    get_precision_at_k_by_k_heads,
    get_gt_heads_order,
)

vit_config = config["vit"]
loss_config = vit_config["loss"]

seed_everything(config["general"]["seed"])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-for-dino",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)
vit_unfreezed = handle_model_config_and_freezing_for_task(
    model=load_ViTModel(vit_config, model_type="vit-for-dino-grad"), freezing_transformer=False
)
n_heads = 12


def optimize_params(
    vit_model: ViTForImageClassification, vit_ours_model, feature_extractor, criterion: Callable
):
    df = pd.DataFrame(
        columns=[
            "grad_min_v",
            "grad_max_v",
            "grad_mean_v",
            "grad_median_v",
            "relu_grad_max_v",
            "relu_grad_mean_v",
            "relu_grad_median_v",
        ]
    )
    predicted_change_df = pd.DataFrame(columns=[""])

    for idx, image_dict in enumerate(vit_config["images"]):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
        image_plot_folder_path = get_and_create_image_plot_folder_path(
            images_folder_path=IMAGES_FOLDER_PATH,
            experiment_name=experiment_name,
            image_name=image_name,
            save_image=False,
        )

        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(
            image_name=image_name, feature_extractor=feature_extractor
        )
        # target = vit_model(**inputs)
        target = vit_unfreezed(**inputs)
        target_class_idx = torch.argmax(target.logits[0])
        loss = objective_grad_rollout(output=target.logits, target_idx=target_class_idx)
        loss.backward()
        attention_probs = get_attention_probs(model=vit_unfreezed)
        gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=False)
        relu_gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=True)

        cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_unfreezed)
        objects_path = create_folder(Path(image_plot_folder_path, "objects"))
        context_layer_grad = (
            vit_unfreezed.vit.encoder.layer[-1]
            .attention.attention.context_layer.grad[0, 0]
            .reshape(n_heads, -1)
        )
        # for head_idx in range(n_heads):
        #     visu(original_image=original_transformed_image,
        #          transformer_attribution=cls_attentions_probs[head_idx],
        #          file_name=Path(image_plot_folder_path, f'head_{head_idx}'))

        resized_image = transforms.PILToTensor()(transforms.Resize((224, 224))(image))
        if not os.path.isfile(Path(objects_path, "gt_heads_order.pkl")):
            change_predicted_class_by_head = eval(
                experiment_dir=image_plot_folder_path,
                model=vit_unfreezed,
                feature_extractor=feature_extractor,
                heads_masks=cls_attentions_probs,
                image=resized_image,
            )
            gt_order_heads = get_gt_heads_order(
                change_predicted_class_by_head=change_predicted_class_by_head
            )
        else:
            gt_order_heads = load_obj_from_path(Path(objects_path, "gt_heads_order.pkl"))
        # grad_min_v = context_layer_grad.min(dim=1)[0]
        # grad_mean_v = context_layer_grad.mean(dim=1)
        # grad_median_v = context_layer_grad.median(dim=1)[0]
        # relu_grad_median_v = F.relu(context_layer_grad).median(dim=1)[0]
        # relu_grad_max_v = F.relu(context_layer_grad).max(dim=1)[0]
        grad_max_v = context_layer_grad.max(dim=1)[0]
        relu_grad_mean_v = F.relu(context_layer_grad).mean(dim=1)

        grad_max_v_order = torch.topk(grad_max_v, n_heads, largest=True)[1].tolist()
        relu_grad_mean_v_order = torch.topk(relu_grad_mean_v, n_heads, largest=True)[1].tolist()

        # attn = list(np.array(attention_probs)[gt_order_heads[:6]])
        # att_grads = list(np.array(gradients)[gt_order_heads[:6]])
        attn = [attn[:, gt_order_heads[:6], :, :] for attn in attention_probs]
        att_grads = [grads[:, gt_order_heads[:6], :, :] for grads in gradients]
        att_relu_grads = [grads[:, gt_order_heads[:6], :, :] for grads in gradients]
        rollout_max_grad_6_heads = rollout(attentions=attn, head_fusion="max", gradients=att_grads)
        rollout_mean_relu_grad_6_heads = rollout(
            attentions=attn, head_fusion="mean", gradients=att_relu_grads
        )
        rollout_max_grad_12_heads = rollout(
            attentions=attention_probs, head_fusion="max", gradients=gradients
        )
        rollout_mean_relu_grad_12_heads = rollout(
            attentions=attention_probs, head_fusion="mean", gradients=relu_gradients
        )
        rollout_mean_grad_12_heads = rollout(
            attentions=attention_probs, head_fusion="mean", gradients=gradients
        )
        rollout_mean_grad_6_heads = rollout(
            attentions=attn, head_fusion="mean", gradients=att_grads
        )

        # grad_max_v_precision_at_k = get_precision_at_k_by_k_heads(gt_order_heads=gt_order_heads,
        #                                                           heads_order=grad_max_v_order)
        # relu_grad_mean_v_precision_at_k = get_precision_at_k_by_k_heads(gt_order_heads=gt_order_heads,
        #                                                                 heads_order=relu_grad_mean_v_order)
        # grad_min_v_precision_at_k = get_precision_at_k_by_k_heads(
        # gt_order_heads=gt_order_heads,
        # heads_order=torch.topk(grad_min_v, n_heads, largest=True)[1].tolist())
        # grad_mean_v_precision_at_k = get_precision_at_k_by_k_heads(
        #     gt_order_heads=gt_order_heads,
        #     heads_order=torch.topk(grad_mean_v, n_heads, largest=True)[1].tolist())
        # grad_median_v_precision_at_k = get_precision_at_k_by_k_heads(
        #     gt_order_heads=gt_order_heads,
        #     heads_order=torch.topk(grad_median_v, n_heads, largest=True)[1].tolist())
        # relu_grad_max_v_precision_at_k = get_precision_at_k_by_k_heads(
        #     gt_order_heads=gt_order_heads,
        #     heads_order=torch.topk(relu_grad_max_v, n_heads, largest=True)[1].tolist())
        # relu_grad_median_v_precision_at_k = get_precision_at_k_by_k_heads(
        #     gt_order_heads=gt_order_heads,
        #     heads_order=torch.topk(relu_grad_median_v, n_heads, largest=True)[1].tolist())

        """
        Check aggregation of heads for masks
        """
        change_predicted_mean_grad_rollout_12_heads_iter = eval(
            experiment_dir=image_plot_folder_path,
            model=vit_unfreezed,
            feature_extractor=feature_extractor,
            heads_masks=rollout_mean_grad_12_heads.reshape(-1),
            image=resized_image,
        )[0]
        change_predicted_mean_grad_rollout_6_heads_iter = eval(
            experiment_dir=image_plot_folder_path,
            model=vit_unfreezed,
            feature_extractor=feature_extractor,
            heads_masks=rollout_mean_grad_6_heads.reshape(-1),
            image=resized_image,
        )[0]
        change_predicted_max_grad_rollout_6_heads_iter = eval(
            experiment_dir=image_plot_folder_path,
            model=vit_unfreezed,
            feature_extractor=feature_extractor,
            heads_masks=rollout_max_grad_6_heads.reshape(-1),
            image=resized_image,
        )[0]

        change_predicted_max_grad_rollout_12_heads_iter = eval(
            experiment_dir=image_plot_folder_path,
            model=vit_unfreezed,
            feature_extractor=feature_extractor,
            heads_masks=rollout_max_grad_12_heads.reshape(-1),
            image=resized_image,
        )[0]
        change_predicted_mean_relu_grad_rollout_6_heads_iter = eval(
            experiment_dir=image_plot_folder_path,
            model=vit_unfreezed,
            feature_extractor=feature_extractor,
            heads_masks=rollout_mean_relu_grad_6_heads.reshape(-1),
            image=resized_image,
        )[0]

        change_predicted_mean_relu_grad_rollout_12_heads_iter = eval(
            experiment_dir=image_plot_folder_path,
            model=vit_unfreezed,
            feature_extractor=feature_extractor,
            heads_masks=rollout_mean_relu_grad_12_heads.reshape(-1),
            image=resized_image,
        )[0]
        # change_predicted_median_cls_iter = eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #                                         feature_extractor=feature_extractor,
        #                                         heads_masks=cls_attentions_probs.median(dim=0)[0],
        #                                         image=resized_image)[0]
        # change_predicted_mean_cls_iter = eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #                                       feature_extractor=feature_extractor,
        #                                       heads_masks=cls_attentions_probs.mean(dim=0),
        #                                       image=resized_image)[0]
        # change_predicted_gt_top_6_median_cls_iter = eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #                                                  feature_extractor=feature_extractor,
        #                                                  heads_masks=
        #                                                  cls_attentions_probs[gt_order_heads[:6]].median(dim=0)[0],
        #                                                  image=resized_image)[0]
        # change_predicted_gt_top_6_mean_cls_iter = eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #                                                feature_extractor=feature_extractor,
        #                                                heads_masks=cls_attentions_probs[gt_order_heads[:6]].mean(dim=0),
        #                                                image=resized_image)[0]
        #
        # change_predicted_relu_mean_mean_agg_top_6_cls_iter = \
        #     eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #          feature_extractor=feature_extractor,
        #          heads_masks=cls_attentions_probs[relu_grad_mean_v_order[:6]].mean(dim=0),
        #          image=resized_image)[0]
        # change_predicted_relu_mean_median_agg_top_6_cls_iter = \
        #     eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #          feature_extractor=feature_extractor,
        #          heads_masks=cls_attentions_probs[relu_grad_mean_v_order[:6]].median(dim=0)[0],
        #          image=resized_image)[0]
        # change_predicted_grad_max_mean_agg_top_6_cls_iter = \
        #     eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #          feature_extractor=feature_extractor,
        #          heads_masks=cls_attentions_probs[grad_max_v_order[:6]].mean(dim=0),
        #          image=resized_image)[0]
        # change_predicted_grad_max_median_agg_top_6_cls_iter = \
        #     eval(experiment_dir=image_plot_folder_path, model=vit_unfreezed,
        #          feature_extractor=feature_extractor,
        #          heads_masks=cls_attentions_probs[grad_max_v_order[:6]].median(dim=0)[0],
        #          image=resized_image)[0]
        predicted_change_df = predicted_change_df.append(
            {
                "image": image_name,
                # 'mean_cls': change_predicted_mean_cls_iter,
                # 'median_cls': change_predicted_median_cls_iter,
                # 'gt_median_agg_top_6_cls': change_predicted_gt_top_6_median_cls_iter,
                # 'gt_mean_agg_top_6_cls': change_predicted_gt_top_6_mean_cls_iter,
                # 'relu_mean_mean_agg_top_6_cls': change_predicted_relu_mean_mean_agg_top_6_cls_iter,
                # 'relu_mean_median_agg_top_6_cls': change_predicted_relu_mean_median_agg_top_6_cls_iter,
                # 'grad_max_mean_agg_top_6_cls': change_predicted_grad_max_mean_agg_top_6_cls_iter,
                # 'grad_max_median_agg_top_6_cls': change_predicted_grad_max_median_agg_top_6_cls_iter,
                "mean_grad_rollout_12_heads": change_predicted_mean_grad_rollout_12_heads_iter,
                "mean_grad_rollout_6_heads": change_predicted_mean_grad_rollout_6_heads_iter,
                "max_grad_rollout_6_heads": change_predicted_max_grad_rollout_6_heads_iter,
                "max_grad_rollout_12_heads": change_predicted_max_grad_rollout_12_heads_iter,
                "mean_relu_grad_rollout_6_heads": change_predicted_mean_relu_grad_rollout_6_heads_iter,
                "mean_relu_grad_rollout_12_heads": change_predicted_mean_relu_grad_rollout_12_heads_iter,
            },
            ignore_index=True,
        )
        visu(
            original_image=original_transformed_image,
            transformer_attribution=rollout_max_grad_6_heads,
            file_name=Path(image_plot_folder_path, f"rollout_max_6_heads"),
        )

        visu(
            original_image=original_transformed_image,
            transformer_attribution=rollout_max_grad_12_heads,
            file_name=Path(image_plot_folder_path, f"rollout_max_12_heads"),
        )

        visu(
            original_image=original_transformed_image,
            transformer_attribution=rollout_mean_grad_6_heads,
            file_name=Path(image_plot_folder_path, f"rollout_mean_6_heads"),
        )
        visu(
            original_image=original_transformed_image,
            transformer_attribution=rollout_mean_grad_12_heads,
            file_name=Path(image_plot_folder_path, f"rollout_mean_12_heads"),
        )

        visu(
            original_image=original_transformed_image,
            transformer_attribution=rollout_mean_relu_grad_6_heads,
            file_name=Path(image_plot_folder_path, f"rollout_mean_relu_6_heads"),
        )

        visu(
            original_image=original_transformed_image,
            transformer_attribution=rollout_mean_relu_grad_12_heads,
            file_name=Path(image_plot_folder_path, f"rollout_mean_relu_12_heads"),
        )

        # visu(original_image=original_transformed_image,
        #      transformer_attribution=cls_attentions_probs.mean(dim=0),
        #      file_name=Path(image_plot_folder_path, f'cls_mean'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=cls_attentions_probs.median(dim=0)[0],
        #      file_name=Path(image_plot_folder_path, f'cls_median'))
        #
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=cls_attentions_probs[grad_max_v_order[:6]].median(dim=0)[0],
        #      file_name=Path(image_plot_folder_path, f'grad_max_top_6_median_agg'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=cls_attentions_probs[grad_max_v_order[:6]].mean(dim=0),
        #      file_name=Path(image_plot_folder_path, f'grad_max_top_6_mean_agg'))
        #
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=cls_attentions_probs[relu_grad_mean_v_order[:6]].median(dim=0)[0],
        #      file_name=Path(image_plot_folder_path, f'relu_grad_mean_top_6_median_agg'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=cls_attentions_probs[relu_grad_mean_v_order[:6]].mean(dim=0),
        #      file_name=Path(image_plot_folder_path, f'relu_grad_mean_top_6_mean_agg'))

        plot_heads_scores(
            image_name=image_name,
            scores=torch.topk(relu_grad_mean_v, n_heads, largest=True)[0].tolist(),
            rank_method="relu_grad_mean_v",
            path=image_plot_folder_path,
        )
        plot_heads_scores(
            image_name=image_name,
            scores=torch.topk(grad_max_v, n_heads, largest=True)[0].tolist(),
            rank_method="grad_max_v",
            path=image_plot_folder_path,
        )
        save_obj_to_disk(path=Path(objects_path, "gt_heads_order"), obj=gt_order_heads)
        save_obj_to_disk(
            path=Path(objects_path, "relu_grad_mean_v_order"), obj=relu_grad_mean_v_order
        )
        save_obj_to_disk(path=Path(objects_path, "grad_max_v_order"), obj=grad_max_v_order)
        save_obj_to_disk(Path(objects_path, "cls_attentions_probs.pkl"), cls_attentions_probs)

        # df = df.append(
        #     {'image': image_name,
        #      # 'grad_min_v': grad_min_v_precision_at_k,
        #      'grad_max_v': grad_max_v_precision_at_k,
        #      # 'grad_mean_v': grad_mean_v_precision_at_k,
        #      # 'grad_median_v': grad_median_v_precision_at_k,
        #      # 'relu_grad_max_v': relu_grad_max_v_precision_at_k,
        #      'relu_grad_mean_v': relu_grad_mean_v_precision_at_k,
        #      # 'relu_grad_median_v': relu_grad_median_v_precision_at_k,
        #      }, ignore_index=True)
        # df.to_csv(Path(PLOTS_PATH, experiment_name, 'precision_at_k.csv'), index=False)
        predicted_change_df.to_csv(
            Path(PLOTS_PATH, experiment_name, "predicted_change_df.csv"), index=False
        )


def plot_heads_scores(image_name: str, scores: List[float], rank_method: str, path: Path):
    plt.plot(np.array(scores))
    plt.title(f'{image_name.replace(".JPEG", "")} - Heads scores - {rank_method}')
    plt.xlabel("Head")
    plt.ylabel("Head Score")
    plt.savefig(fname=Path(path, f"head_scores_{rank_method}.png"), format="png")
    plt.show()


if __name__ == "__main__":
    experiment_name = f"heads_context_vector"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    vit_ours_model, optimizer = setup_model_and_optimizer(model_name="softmax_temp")
    optimize_params(
        vit_model=vit_model,
        vit_ours_model=vit_ours_model,
        feature_extractor=feature_extractor,
        criterion=objective_temp_softmax,
    )
