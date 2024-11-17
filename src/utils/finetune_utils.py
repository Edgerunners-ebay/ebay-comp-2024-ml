import os
from typing import List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torchtune.training.memory import OptimizerInBackwardWrapper


def get_lr(
    optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
) -> float:
    """
    Full_finetune_distributed and full_finetune_single_device assume all optimizers have
    the same LR, here to validate whether all the LR are the same and return if True.

    Args:
        optimizer (Union[torch.optim.Optimizer, OptimizerInBackwardWrapper]): A general
            optimizer input that could whether be a general optimizer or an optimizer
            warpper based on optimizer_in_backward.

    Returns:
        lr (float): The learning rate of the input optimizers.

    Raises:
        RuntimeError: If the learning rates of the input optimizer are not the same.
    """
    if isinstance(optimizer, OptimizerInBackwardWrapper):
        param_groups = []
        for param in optimizer.state_dict().values():
            param_groups.append(param["param_groups"][0])
    else:
        param_groups = optimizer.param_groups
    if len(param_groups) < 1:
        raise RuntimeError(
            f"Invalid optimizer param groups with len of: {len(param_groups)}"
        )

    # LR Schedulers are the same across all param groups for full_finetune right now
    lr = param_groups[0]["lr"]
    for group in param_groups:
        if group["lr"] != lr:
            raise RuntimeError("LR Schedulers are different across all param groups ")
    return lr


def get_shard_conditions(
    name: str,
    module: nn.Module,
    names_to_match: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> bool:
    """
    Returs True for layers named {}.layers.i or layers that exactly match names_to_match, otherwise,
    returns False. This is a helper function for sharding a model with FSDP.
    In :func:`~torchtune.training.shard_model`, we iterate over the model's named modules
    and apply fully_shard using this condition.

    As part of our sharding strategy, we want each layer to be sharded separately, as this is
    generally efficient. We may also want to shard certain modules that are not layers, such as
    the embedding module.

    #TODO: a more robust way would be to shard on the module type, not the name.

    Args:
        name (str): Name of the module.
        module (nn.Module): Module to be sharded.
        names_to_match (Optional[List[str]]): List of names to match, if any.
        *args: Variable length argument list to be passed to the Embedding module.
        **kwargs: Arbitrary keyword arguments to be passed to the Embedding module.

    Returns:
        bool: True if the module name matches the condition, False otherwise.

    Examples:
        >>> names_to_match = ["embedding"]
        >>> layer_names = ["layers.0", "decoder.layers.1", "encoder.layers.2.attention",
            "my_wrapper.layer.1.something", "embedding"]
        >>> matches = []
        >>> for name in layer_names:
        >>>     if shard_condition_is_layer_or_match(name, None): matches.append(name)
        >>> print(matches)
        >>> ["layers.0", "decoder.layers.1", "embedding"]
    """
    if names_to_match and name in names_to_match:
        return True

    name_list = name.split(".")
    if len(name_list) >= 2:
        return name_list[-2] == "layers" and str.isdigit(name_list[-1])

    return False
