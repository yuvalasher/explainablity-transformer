import argparse
from distutils.util import strtobool
from utils.consts import MODEL_OPTIONS

def get_parser(params_config):
    """
    Create and configure an argument parser for the pLTX model training script.

    This function sets up an argparse.ArgumentParser with various command-line arguments
    for configuring model architecture, training parameters, data handling, and other
    options for the pLTX model training process.

    Args:
        params_config (dict): A dictionary containing default values for various parameters.

    Returns:
        argparse.ArgumentParser: A configured argument parser object with all the
        necessary arguments for training the pLTX model.

    The parser includes arguments for:
    - Model configuration (e.g., model names, activation function, image size)
    - Training parameters (e.g., number of epochs, batch size, learning rate)
    - Data handling (e.g., number of samples per label for training and validation)
    - Various boolean flags for controlling model behavior
    - Other miscellaneous options (e.g., paths for plots and checkpoints)

    Most arguments have default values taken from the params_config dictionary.
    """
    parser = argparse.ArgumentParser(description='Train pLTX model')

    # Model arguments
    parser.add_argument('--explainer-model-name', type=str, default="vit_base_224", choices=MODEL_OPTIONS)
    parser.add_argument('--explainee-model-name', type=str, default="vit_base_224", choices=MODEL_OPTIONS)
    parser.add_argument('--activation-function', type=str, default=params_config["activation_function"])
    parser.add_argument('--img-size', type=int, default=params_config["img_size"])
    parser.add_argument('--patch-size', type=int, default=params_config["patch_size"])

    # Training arguments
    parser.add_argument('--n-epochs', type=int, default=params_config["n_epochs"])
    parser.add_argument('--batch-size', type=int, default=params_config["batch_size"])
    parser.add_argument('--lr', type=float, default=params_config["lr"])
    parser.add_argument('--mask-loss', type=str, default=params_config["mask_loss"])
    parser.add_argument('--mask-loss-mul', type=int, default=params_config["mask_loss_mul"])
    parser.add_argument('--prediction-loss-mul', type=int, default=params_config["prediction_loss_mul"])
    parser.add_argument('--prediction-neg-loss-mul', type=int, default=params_config["prediction_neg_loss_mul"])

    # Data arguments
    parser.add_argument('--train-n-label-sample', type=int, default=params_config["train_n_label_sample"])
    parser.add_argument('--val-n-label-sample', type=int, default=params_config["val_n_label_sample"])

    # Boolean arguments
    bool_args = [
        ("train-model-by-target-gt-class", "train_model_by_target_gt_class"),
        ("enable-checkpointing", "enable_checkpointing"),
        ("verbose", "verbose"),
        ("is-sampled-train-data-uniformly", "is_sampled_train_data_uniformly"),
        ("is-sampled-val-data-uniformly", "is_sampled_val_data_uniformly"),
        ("is-freezing-explaniee-model", "is_freezing_explaniee_model"),
        ("is-clamp-between-0-to-1", "is_clamp_between_0_to_1"),
        ("is-competitive-method-transforms", "is_competitive_method_transforms"),
        ("use-logits-only", "use_logits_only"),
    ]
