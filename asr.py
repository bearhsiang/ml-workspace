import hydra
from omegaconf import DictConfig, OmegaConf
import main.train as train
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="asr.yaml")
def main(config : DictConfig):
    if config.mode == 'train':
        train(config)

if __name__ == '__main__':
    main()