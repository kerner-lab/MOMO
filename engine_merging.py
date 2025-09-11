
import os

import torch
from typing import Dict
from task_vectors import TaskVector, merge_max_abs

def create_combined_encoder(
    model_combination: Dict[str, str],
    pre_trained_model: str,
    which_merging_technique: str,
    output_dir: str,
    device: torch.device,
    scaling_coef: float = 0.5
) -> torch.nn.Module:
    """
    Create a combined encoder using task vectors.
    
    Args:
        model_combination: Dictionary mapping model names to checkpoint epochs
        pre_trained_model: Path to pre-trained model
        which_merging_technique: Merging technique ('task_vectors' or 'magmax')
        output_dir: Base output directory
        device: Device to create model on
        scaling_coef: Scaling coefficient for task vector application
        
    Returns:
        Combined encoder model
    """
    # Create task vectors
    task_vectors = []
    for model_name, checkpoint in model_combination.items():
        model_type = model_name.split("_")[-1]
        model_prefix = "_".join(model_name.split("_")[:-1])
        checkpoint_path = os.path.join(
            output_dir, "pretraining", model_type, model_prefix, 
            f"encoder_epoch_{checkpoint}.pth"
        )
        task_vectors.append(TaskVector(pre_trained_model, checkpoint_path))

    # Merge task vectors
    if which_merging_technique == "task_vectors":
        task_vector_sum = sum(task_vectors)
    elif which_merging_technique == "magmax":
        task_vector_sum = merge_max_abs(task_vectors)
    else:
        raise ValueError(f"Unknown merging technique: {which_merging_technique}. "
                        f"Available choices: task_vectors, magmax")

    # Apply task vector to pre-trained model
    model_type = list(model_combination.keys())[0].split("_")[-1]
    combined_encoder = task_vector_sum.apply_to(
        pre_trained_model, model_type, device, scaling_coef=scaling_coef
    ).to(device)

    # Save combined model
    combined_model_name = "_".join([f"{model}_{checkpoint}" for model, checkpoint in model_combination.items()])
    save_dir = os.path.join(output_dir, f"combined_models_{which_merging_technique}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{combined_model_name}.pth")
    print(f"Saving combined encoder to {save_path}")
    torch.save(combined_encoder.state_dict(), save_path)

    return combined_encoder
