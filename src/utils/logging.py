import wandb

def log_metrics(metrics, epoch, step, run, cfg):
        run.log({
            "epoch": epoch,
            "step": step,
            **metrics
        })
