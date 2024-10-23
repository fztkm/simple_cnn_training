import argparse
import os


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR


from utils import compute_topk_accuracy, log_params_on_step
from model import configure_model, ModelConfig
from setup import (
    configure_optimizer,
    configure_warmup_cosine_decay_lambda,
    configure_constant_lambda,
)


def get_grouped_params(
    model,
    no_decay=["bias", "LayerNorm.weight", "layer_norm.weight"],
    head=["classifier"],
    base_lr=1e-5,
    head_lr=1e-2,
    weight_decay=5e-4,
):
    base_params_with_wd, base_params_without_wd = [], []
    head_params_with_wd, head_params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in head):
            if any(nd in n for nd in no_decay):
                head_params_without_wd.append(p)
            else:
                head_params_with_wd.append(p)
        elif any(nd in n for nd in no_decay):
            base_params_without_wd.append(p)
        else:
            base_params_with_wd.append(p)

    return [
        {"params": base_params_with_wd, "weight_decay": weight_decay, "lr": base_lr},
        {"params": base_params_without_wd, "weight_decay": 0.0, "lr": base_lr},
        {"params": head_params_with_wd, "weight_decay": weight_decay, "lr": head_lr},
        {"params": head_params_without_wd, "weight_decay": 0.0, "lr": head_lr},
    ]


class SimpleLightningModel(pl.LightningModule):
    """A simple lightning module

        see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#methods
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#properties
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#method-order
    """

    def __init__(
            self,
            command_line_args: argparse.Namespace,
            n_classes: int,
            exp_name: str,
            warmup_rate: float = 0.1,
    ):
        """constructor

        see
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#init

        Args:
            command_line_args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
        """
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name
        self.warmup_rate = warmup_rate
        self.current_step_for_scheduler = 0


        self.model = configure_model(ModelConfig(
            model_name=self.args.model_name,
            use_pretrained=self.args.use_pretrained,
            torch_home=self.args.torch_home,
            n_classes=n_classes,
        ))

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            # model_params=self.model.parameters()
            model_params=get_grouped_params(self.model, base_lr=self.args.lr)
        )
        max_steps = self.trainer.estimated_stepping_batches
        warmup_step = int(max_steps * self.warmup_rate)
        backbone_lambda = configure_warmup_cosine_decay_lambda(max_steps, warmup_step)
        head_lambda = configure_warmup_cosine_decay_lambda(max_steps, 0)
        # head_lambda = configure_constant_lambda()

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[backbone_lambda, backbone_lambda, head_lambda, head_lambda],)

        return {
            "optimizer": optimizer,
            "lr_scheduler":
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def configure_callbacks(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="val_top1",
                mode="max",  # larger is better
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={val_top1:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def log_train_loss_top15(self, loss, top1, top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_top1": top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )


    def training_step(self, batch, batch_idx):
        """a single training step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Returns:
            tensor: loss (used for backward by lightning)

        Note:
            DO NOT USE .to() or model.train() here
                (automatically send to multi-GPUs)
            DO NOT USE loss.backward() here
                (automatically performed by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data, labels=labels)
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_train_loss_top15(loss, top1, top5, batch_size)

        log_params_on_step(self, {
            "lr_backbone": self.trainer.optimizers[0].param_groups[0]["lr"],
            "lr_head": self.trainer.optimizers[0].param_groups[2]["lr"],
        }, batch_size)

        return loss

    def log_val_loss_top15(self, loss, top1, top5, batch_size):
        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        """a single validation step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Note:
            DO NOT USE .to() or model.eval() here
                (automatically send to multi-GPUs)
            DO NOT USE with torch.no_grad() here
                (automatically handled by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data, labels=labels)
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_val_loss_top15(loss, top1, top5, batch_size)
