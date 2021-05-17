from monitor import WandbMonitor

def get_monitor(type, config):
    if type == 'wandb':
        return WandbMonitor(**config)