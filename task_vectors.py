
import torch
import torch.nn as nn

from models_pretrain import *


class TaskVector():

    @staticmethod
    def _extract_model_state_dict(checkpoint):
        """Extract model state dict from checkpoint based on your saving format."""
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                return checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            else:
                return checkpoint
        else:
            return checkpoint.state_dict()

    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """
        Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                # Load checkpoints and extract model state dicts
                pretrained_loaded = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
                finetuned_loaded = torch.load(finetuned_checkpoint, map_location='cpu', weights_only=False)
 
                # Extract model state dicts based on your checkpoint format
                pretrained_state_dict = self._extract_model_state_dict(pretrained_loaded)
                finetuned_state_dict = self._extract_model_state_dict(finetuned_loaded)
                self.vector = {}
                for key in pretrained_state_dict:
                    # Check if the value is a tensor and has the right dtype
                    if not isinstance(pretrained_state_dict[key], torch.Tensor):
                        continue
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    if key not in finetuned_state_dict:
                        continue
                    if not isinstance(finetuned_state_dict[key], torch.Tensor):
                        continue

                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]


    def __add__(self, other):

        """ Add two task vectors together. """
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    # print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]

        return TaskVector(vector=new_vector)

    def __radd__(self, other):

        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)


    def __neg__(self):

        """ Negate a task vector. """
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]

        return TaskVector(vector=new_vector)


    def apply_to(self, pretrained_checkpoint, train_model, device, args, scaling_coef=1.0):

        """ Apply a task vector to a pretrained model. """
        with torch.no_grad():
            pretrained_model = create_model(
                train_model=train_model,
                model_unit="encoder",
                device=device,
                if_pretrained=False,
                args=args
            )
            # Load checkpoint and extract model state dict
            pretrained_loaded = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
            pretrained_state_dict = self._extract_model_state_dict(pretrained_loaded)
            new_state_dict = {}
            for key in pretrained_state_dict:
                if key not in self.vector:
                    # print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue

                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)

        return pretrained_model


def merge_max_abs(task_vectors):
    """ Mix multiple task vectors together by highest parameter value. """
    if len(task_vectors) == 0:
        return None

    if len(task_vectors) == 1:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}

        # Iterate over keys in the first task vector
        for key in task_vectors[0].vector:
            # Get the initial tensor for the current key
            max_abs_tensor = task_vectors[0].vector[key]

            # Iterate over the remaining task vectors
            for task_vector in task_vectors[1:]:
                current_tensor = task_vector.vector[key]
                
                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(current_tensor.abs() >= max_abs_tensor.abs(), current_tensor, max_abs_tensor)
            
            # Assign the final tensor to the new_vector dictionary
            new_vector[key] = max_abs_tensor

    return TaskVector(vector=new_vector)
