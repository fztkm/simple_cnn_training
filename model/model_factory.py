import os

import torch

from model import (
    ClassificationBaseModel,
    ModelConfig,
)

from model.resnet import (
    ResNet18,
    ResNet50,
)
from model.vit import ViTb
from model.clip import CLIPImageClassifier


def set_torch_home(
    model_info: ModelConfig
) -> None:
    """Specity the directory where a pre-trained model is stored.
    Otherwise, by default, models are stored in users home dir `~/.torch`
    """
    os.environ['TORCH_HOME'] = model_info.torch_home


def configure_model(
        model_info: ModelConfig
) -> ClassificationBaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """

    if model_info.use_pretrained:
        set_torch_home(model_info)

    if model_info.model_name == 'resnet18':
        model = ResNet18(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'resnet50':
        model = ResNet50(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'vit_b':
        model = ViTb(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'clip':
        model = CLIPImageClassifier(model_info)

    else:
        raise ValueError('invalid model_info.model_name')

    return model
