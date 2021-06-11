from factory import get_model, get_optimizer, get_dataset, get_monitor
from torch.utils.data import DataLoader
import logging
import os
import torch
from tqdm.auto import tqdm
logger = logging.getLogger(__name__)

class Runner:

    def __init__(self, total_steps, batch_size, grad_accu_step, valid_step, log_step, save_step, n_workers, data):

        self.total_steps = total_steps
        self.batch_size = batch_size
        self.grad_accu_step = grad_accu_step
        self.valid_step = valid_step
        self.save_step = save_step
        self.log_step = log_step
        self.n_workers = n_workers
        self.data = data
        
        self.dataset = {}
        self.state = {
            'step': 0,
        }

    def set_device(self, device):
        self.device = device

    def set_model(self, **config):
        self.model = get_model(**config)
        self.model.to(self.device)
        self.model.zero_grad()
    
    def get_model(self):
        return self.model

    def set_optimizer(self, **config):
        self.optimizer = get_optimizer(self.model.parameters(), **config)
    
    def get_optimizer(self):
        return self.optimizer

    def set_data(self, **config):

        for mode in self.data:
            for split in self.data[mode]:
                if split in self.dataset:
                    continue
                dataset_config = {**config['config'], **config['split'][split]}
                
                self.add_split(split, config['wrapper'], dataset_config)

    def add_split(self, split, wrapper, config):
        self.dataset[split] = get_dataset(wrapper, config)

    def set_monitor(self, **config):
        self.monitor = get_monitor(**config)

    def get_data(self, mode, split):
        return DataLoader(
            self.dataset[split],
            batch_size=self.batch_size[mode],
            num_workers=self.n_workers,
            collate_fn=getattr(self.dataset[split], 'collate_fn', None)
        )

    def get_splits(self, mode):
        return self.data[mode]

    def set_criterion(self, config):
        pass

    def get_state(self):
        return self.state

    def update_state(self, state):
        self.state.update(state)

    def step(self, raw_batch, log, mode):
        
        batch = self.create_batch(raw_batch, mode)
        if mode == 'train':
            model_input, tgt = batch
            logit = self.model(**model_input)
            loss = self.count_loss(logit, tgt) / self.grad_accu_step
            loss.backward()
            log['loss'].append(loss.item())
        else:
            model_input = batch
            output = self.model.inference(**model_input)
            log['output'].append(output)

    def update_model(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def create_batch(raw_batch, mode=False):
        '''
        if mode == 'train'
            return model_input, target
        else
            return model_input
        
        model will consume the model_input directly, so be sure to put them in the same device

        '''
        pass

    def count_loss(self, hypo, gold):
        return self.criterion(hypo, gold)

    def log(self, log, mode):
        self.monitor.log(log)
        
    def load(self):
        if os.path.isfile('model.pth'):
            logger.info('found model checkpoint! load from the checkpoint...')
            self.get_model().load_state_dict(torch.load('model.pth'))
            self.get_model().to(self.device)
        if os.path.isfile('optimizer.pth'):
            logger.info('found optimizer checkpoint! load from the checkpoint...')
            self.get_optimizer().load_state_dict(torch.load('optimizer.pth'))
        if os.path.isfile('trainer.pth'):
            logger.info('found trainer checkpoint! load from the checkpoint...')
            self.state = torch.load('trainer.pth')

    def save(self):
        logger.info(f'step: {self.state["step"]} save checkpoint...')
        torch.save(self.get_model().state_dict(), 'model.pth')
        torch.save(self.get_optimizer().state_dict(), 'optimizer.pth')
        torch.save(self.state_dict(), 'trainer.pth')

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def state_dict(self):
        return self.state

