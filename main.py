from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import torch
import logging
from factory import get_runner


logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="main.yaml")
def main(config : DictConfig):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = get_runner(**config['runner'])
    # runner = TranslationTrainer(**config['runner']['config'])
    runner.set_device(device)
    runner.set_model(**config['model'])
    runner.set_optimizer(**config['optimizer'])
    runner.set_data(**config['dataset'])
    runner.set_criterion()
    runner.set_monitor(**config['monitor'])
    # runner.load()

    training_state = runner.get_state()
    step = training_state['step']
    total_steps = runner.total_steps
    logger.info('start training')


    with tqdm(total=total_steps, initial=step, desc="steps") as step_bar:

        train_step = 0 # number of forward batch (not the update step)

        while step < total_steps:

            for train_split in runner.get_splits(mode='train'):

                train_data = runner.get_data(mode='train', split=train_split)
                train_bar = tqdm(train_data, desc='train')
                train_log = defaultdict(list)

                runner.train()

                for train_batch in train_bar:

                    runner.step(train_batch, log=train_log, mode='train')
                    train_step += 1

                    if train_step % runner.grad_accu_step == 0:
                        runner.update_model()
                        step += 1
                        runner.update_state({
                            'step': step,
                        })
                        step_bar.update(1)

                        if step % runner.log_step == 0:
                            runner.log(train_log, mode='train', split=train_split)
                            train_log = defaultdict(list)

                        if step % runner.valid_step == 0:
                            
                            runner.eval()
                            
                            for valid_split in runner.get_splits(mode='valid'):

                                valid_data = runner.get_data(mode='valid', split=valid_split)
                                valid_bar = tqdm(valid_data, desc=valid_split)
                                valid_log = defaultdict(list)

                                with torch.no_grad():
                                    for valid_batch in valid_bar:
                                        runner.step(valid_batch, log=valid_log, mode='valid')

                                runner.log(valid_log, mode='valid', split=valid_split)

                            runner.train()

                        if step % runner.save_step == 0:
                            runner.save()

if __name__ == '__main__':
    main()