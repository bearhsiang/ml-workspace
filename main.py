from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import TranslationTrainer
from tqdm.auto import tqdm
import torch

@hydra.main(config_path="conf", config_name="translation_trainer.yaml")
def main(config : DictConfig):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # trainer = get_trainer(config['trainer'])
    trainer = TranslationTrainer(**config['trainer']['config'])
    trainer.set_device(device)
    trainer.set_model(**config['model'])
    trainer.set_optimizer(**config['optimizer'])
    trainer.set_data(**config['data'])
    trainer.set_criterion()
    trainer.set_monitor(**config['monitor'])

    training_state = trainer.get_state()
    epoch = training_state['epoch']
    step = training_state['inner_step']//trainer.batch_size['train']

    epoch_bar = tqdm(range(trainer.total_epochs), initial=epoch, desc="epoch")

    for epoch in epoch_bar:

        train_data = trainer.get_data(split='train')    
        train_bar = tqdm(train_data, initial=step, desc='train')
        train_log = defaultdict(list)

        for train_batch in train_bar:

            trainer.step(train_batch, log=train_log, mode='train')

            if (step+1) % trainer.grad_accu_step == 0:
                trainer.update_model()
                trainer.update_state({
                    'inner_step': step*trainer.batch_size['train'],
                    'epoch': epoch,
                })

            if (step+1) % trainer.log_step == 0:
                trainer.log(train_log, mode='train', split='train')
                train_log = defaultdict(list)

            if (step+1) % trainer.valid_step == 0:
                
                for split in trainer.get_splits(mode='valid'):

                    valid_data = trainer.get_data(split=split)
                    valid_bar = tqdm(valid_data, desc='split')
                    valid_log = defaultdict(list)

                    with torch.no_grad():
                        for valid_batch in valid_bar:
                            trainer.step(valid_batch, log=valid_log, mode='valid')

                    trainer.log(valid_log, mode='valid', split=split)

            if (step+1) % trainer.save_step == 0:
                trainer.save()

            step += 1
        
        step = 0
        


        




if __name__ == '__main__':
    main()