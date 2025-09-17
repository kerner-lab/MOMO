
import os
import argparse

import torch
from typing import Dict

from task_vectors import TaskVector, merge_max_abs
import utils.misc as misc


def create_combined_encoder(
    model_combination: Dict[str, str],
    train_model: str,
    pretrained_model_path: str,
    which_merging_technique: str,
    output_dir: str,
    suffix: str,
    device: torch.device,
    scaling_coef: float,
    args: argparse.Namespace
) -> torch.nn.Module:
    """
    Create a combined encoder using task vectors.
    
    Args:
        model_combination: Dictionary mapping model names to checkpoint epochs
        pretrained_model_path: Path to pre-trained model
        which_merging_technique: Merging technique ('task_vectors' or 'magmax')
        output_dir: Base output directory
        suffix: Suffix to add to the combined model name
        device: Device to create model on
        scaling_coef: Scaling coefficient for task vector application
        
    Returns:
        Combined encoder model
    """
    # Create task vectors
    task_vectors = []
    for instrument, checkpoint in model_combination.items():
        checkpoint_path = os.path.join(
            output_dir, "pretraining", train_model, instrument, 
            f"{instrument}-{checkpoint}.pth"
        )
        task_vectors.append(TaskVector(pretrained_model_path, checkpoint_path))

    # Merge task vectors
    if which_merging_technique == "task_vectors":
        task_vector_sum = sum(task_vectors)
    elif which_merging_technique == "magmax":
        task_vector_sum = merge_max_abs(task_vectors)
    else:
        raise ValueError(f"Unknown merging technique: {which_merging_technique}. "
                        f"Available choices: task_vectors, magmax")

    # Apply task vector to pre-trained model
    combined_encoder = task_vector_sum.apply_to(
        pretrained_model_path, train_model, device, scaling_coef=scaling_coef
    ).to(device)

    # Save combined model
    combined_model_name = "_".join([f"{instrument}" for instrument, checkpoint in model_combination.items()])
    save_dir = os.path.join(output_dir, "pretraining", train_model, f"model_merging_{which_merging_technique}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving combined encoder to {os.path.join(save_dir, f'{combined_model_name}_{suffix}.pth')}")

    if "vit" in train_model:
        misc.save_model(args, save_dir, f"{combined_model_name}_{suffix}", combined_encoder)
    else:
        torch.save(combined_encoder.state_dict(), os.path.join(save_dir, f"{combined_model_name}_{suffix}.pth"))

    return combined_encoder
