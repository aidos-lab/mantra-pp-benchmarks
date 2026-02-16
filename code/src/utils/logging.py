import wandb

def log_metric(name, value, job_type, epoch, step, cfg):
    with wandb.init(project=cfg.logger.project, job_type=job_type) as run:
        run.log({
            "epoch": epoch,
            "step": step,
            name: value,
        })
