from .WandbMonitor import *

def get_monitor(type, config):
    if type == 'wandb':
        return WandbMonitor(**config)