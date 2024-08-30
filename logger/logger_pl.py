from typing import Optional, List, Tuple

from comet_ml import Experiment
from lightning.pytorch.loggers import CometLogger


def configure_logger(
        disable_logging: bool,
        save_dir: str,
        exp_name: str,
        tags: List,
        existing_exp_key: Optional[str] = None,
) -> Tuple[Experiment, str]:
    """comet logger factory

    Args:
        model_name (str): modelname to be added as a tag of comet experiment
        disable_logging (bool): disable comet Experiment object
        save_dir (str): dir to save comet log

    Returns:
        comet_ml.Experiment: logger
        str: experiment name of comet.ml
    """
    if existing_exp_key:
        exp_name, tags = None, None

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    comet_logger = CometLogger(
        save_dir=save_dir,
        experiment_name=exp_name,
        parse_args=True,
        disabled=disable_logging,
    )

    if tags:
        comet_logger.experiment.add_tags(tags)

    return comet_logger, exp_name
