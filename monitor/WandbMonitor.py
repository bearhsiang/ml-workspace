from .Monitor import Monitor
import wandb

class WandbMonitor(Monitor):

    def __init__(self, **config):
        super().__init__()
        self.run = wandb.init(**config)

    def trace(self, trainer):
        self.run.watch(trainer.get_model)

    def log(self, log):
        self.run.log(log)