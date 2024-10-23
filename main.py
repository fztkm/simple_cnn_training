
import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm


from args import ArgParse
from logger import configure_logger, create_exp_name
from callback import configure_callbacks
from dataset import TrainValDataModule
from model import SimpleLightningModel


def main():
    assert torch.cuda.is_available()

    args = ArgParse.get()

    experiment_key = args.comet_exp_key
    if experiment_key:
        exp_name, tags = None, None
    else:
        exp_name, tags = create_exp_name(args)

    loggers, exp_name = configure_logger(
        disable_logging=args.disable_comet,
        save_dir=args.comet_log_dir,
        exp_name=exp_name,
        tags=tags,
        existing_exp_key=experiment_key,
    )

    data_module = TrainValDataModule(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )
    max_steps = len(data_module.train_dataloader()) * args.num_epochs // args.grad_accum

    model_lightning = SimpleLightningModel(
        command_line_args=args,
        n_classes=data_module.n_classes,
        exp_name=exp_name,
        warmup_rate=0.1,
    )

    callbacks = configure_callbacks()

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    if not args.debug:
        trainer = pl.Trainer(
            devices=args.devices,
            accelerator="gpu",
            strategy="auto",
            max_epochs=args.num_epochs,
            logger=loggers,
            log_every_n_steps=args.log_interval_steps,
            accumulate_grad_batches=args.grad_accum,
            num_sanity_val_steps=0,
            # precision="16-true",  # for FP16 training, use with caution for nan/inf
            # fast_dev_run=True, # only for debug
            # fast_dev_run=5,  # only for debug
            # limit_train_batches=15,  # only for debug
            # limit_val_batches=15,  # only for debug
            callbacks=callbacks,
            plugins=[TorchSyncBatchNorm()],
            # profiler="simple",
        )
    else:
        trainer = pl.Trainer(
            devices=args.devices,
            accelerator="gpu",
            strategy="auto",
            max_epochs=args.num_epochs,
            logger=loggers,
            log_every_n_steps=args.log_interval_steps,
            accumulate_grad_batches=args.grad_accum,
            limit_train_batches=150,  # only for debug
            limit_val_batches=20,  # only for debug
            callbacks=callbacks,
            plugins=[TorchSyncBatchNorm()],
        )

    if args.val_only:
        trainer.validate(
            model=model_lightning,
            datamodule=data_module,
            ckpt_path=args.checkpoint_to_resume,
        )
    else:
        trainer.fit(
            model=model_lightning,
            datamodule=data_module,
            ckpt_path=args.checkpoint_to_resume,
        )


if __name__ == "__main__":
    main()
