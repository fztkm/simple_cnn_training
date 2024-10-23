from lightning import LightningModule


def log_params_on_step_on_epoch(module: LightningModule, params, batch_size, show_on_prog_bar=False):
    module.log_dict(
        params,
        prog_bar=show_on_prog_bar,
        on_step=True,
        on_epoch=True,
        rank_zero_only=False,
        sync_dist=True,
        batch_size=batch_size,
        add_dataloader_idx=False,
    )


def log_params_on_step(module: LightningModule, params, batch_size):
    module.log_dict(
        params,
        prog_bar=False,
        on_step=True,
        on_epoch=False,
        rank_zero_only=False,
        sync_dist=True,
        batch_size=batch_size,
        add_dataloader_idx=False,
    )


def log_train_loss_top15(module: LightningModule, loss, top1, top5, batch_size):
    # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
    log_params_on_step_on_epoch(
        module,
        {"train_loss": loss.item(), "train_top1": top1},
        batch_size,
        show_on_prog_bar=True
    )
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
    log_params_on_step_on_epoch(module, {"train_top5": top5}, batch_size)


def log_val_loss_top15(module: LightningModule, loss, top1, top5, batch_size, suffix=None):
    module.log_dict(
        {
            f"val_loss{suffix}": loss.item(),
            f"val_top1{suffix}": top1,
            f"val_top5{suffix}": top5,
        },
        prog_bar=False,
        on_step=True,
        on_epoch=True,
        rank_zero_only=False,
        sync_dist=True,  # sync log metrics for validation
        batch_size=batch_size,
        add_dataloader_idx=False,
    )
