from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import (
    instantiate_callbacks,
    get_ckpt_path,
    model_config_from_id,
    get_current_time,
)
import pytorch_lightning as pl
import os
import hydra
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Callback
import wandb
import yaml
import logging

os.environ["HYDRA_FULL_ERROR"] = "1"


def update_dict(d: dict | list[dict]) -> None:
    """
    Recursively update the dictionary to replace the PretrainedModel config with the one from the experiment id of the pretrained model.
    Why? Because if the the same model is finetuned multiple times, the initialization process will be a mess since it will load all previous configs.
    """
    if isinstance(d, dict):
        if d.get("_target_", None) == "src.networks.PretrainedModel":
            project = d["project"]
            id = d["id"]
            model_keyword = d["model_keyword"]
            model_config = model_config_from_id(project, id, model_keyword)
            d.clear()
            d.update(model_config)
        for k, v in d.items():
            update_dict(v)
    elif isinstance(d, list):
        for v in d:
            update_dict(v)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)

    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    update_dict(config)

    logger.info("Config:\n%s", yaml.dump(config, default_flow_style=False, sort_keys=False))
    
    if (continue_from_id := cfg.continue_from_id):
        id = continue_from_id
        assert cfg.ckpt_filename is not None, "'ckpt_filename' must be provided when continue_from_id is set."
        ckpt_path = get_ckpt_path(cfg.project_name, id, cfg.ckpt_filename)
        logger.info(f"Continuing from id: {id} \n --> Using checkpoint path: {ckpt_path}")
    else:
        assert cfg.ckpt_filename is None, "'ckpt_filename' should be None when not continuing from an id."
        id, ckpt_path = get_current_time(), None
        
    wandblogger = WandbLogger(
        **cfg.logger,
        project=cfg.project_name,
        name=cfg.task_name,
        id=id,
        config=config,
    )

    logger.info("Instantiating callbacks..")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks", None))

    logger.info("Setting up trainer..")
    trainer = Trainer(**cfg.trainer, logger=wandblogger, callbacks=callbacks)

    logger.info("Instantiating model and datamodule..")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        logger.info("Compiling model..")
        logger.warning("You cannot compile models on CPU. Make sure you are using a GPU!")
        model = torch.compile(model)

    if cfg.phase == "train":
        logger.info("Beginning training..")
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    if cfg.phase == "test":
        logger.info("Beginning testing..")
        trainer.test(model, datamodule, ckpt_path=ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    my_app()
