from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig

from model import ModelConfig, ClassificationBaseModel, ModelOutput


class CLIPImageClassifier(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        # self.prepare_model()
        # self.classifier = torch.nn.Module()
        config = CLIPVisionConfig.from_pretrained(
            "openai/clip-vit-base-patch32",
        )
        # self.model = CLIPVisionModelWithProjection.from_pretrained(
        #     "openai/clip-vit-base-patch32",
        #     config=config
        # ).requires_grad_(False)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32",
            config=config
        ).requires_grad_(True)
        self.layer_norm = torch.nn.LayerNorm(config.projection_dim)
        self.classifier = torch.nn.Linear(
            in_features=config.projection_dim,
            # out_features=self.model_config.n_classes,
            out_features=365,
        )

    # def prepare_model(self):
        # config = CLIPVisionConfig.from_pretrained(
        #     "openai/clip-vit-base-patch32",
        # )
        # self.model = CLIPVisionModelWithProjection.from_pretrained(
        #     "openai/clip-vit-base-patch32",
        #     config=config
        # ).requires_grad_(False)
        # self.classifier = torch.nn.Linear(
        #     in_features=config.projection_dim,
        #     out_features=self.model_config.n_classes,
        # )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        assert isinstance(self.model, CLIPVisionModelWithProjection)
        output = self.model(
            pixel_values=pixel_values,
        )
        embedds = output.image_embeds
        logits = self.classifier(self.layer_norm(embedds))

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(label_smoothing=0.0)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model_config.n_classes), labels.view(-1))

        return ModelOutput(
            logits=logits,
            loss=loss
        )
