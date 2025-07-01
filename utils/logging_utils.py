from icecream import ic


def log_configuration(args, explainer_model, explainee_model, data_path):
    """Log the configuration parameters in a structured manner."""
    ic("Model Configuration:")
    ic(explainer_model, explainee_model)

    ic("Training Parameters:")
    ic(args.batch_size,
       args.train_model_by_target_gt_class,
       args.enable_checkpointing)

    ic("Data Sampling Strategy:")
    ic(args.is_sampled_train_data_uniformly,
       args.is_sampled_val_data_uniformly,
       args.is_competitive_method_transforms)

    ic("Miscellaneous:")
    ic(args.verbose,
       str(data_path))
