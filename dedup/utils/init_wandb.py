import wandb


def init_wandb(cfg):
    return wandb.init(project=cfg.wandb_project, config=cfg.model_dump())
