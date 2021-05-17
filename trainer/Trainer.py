from factory import get_model, get_optimizer, get_dataset, get_monitor
from torch.utils.data import DataLoader

class Trainer:

    def __init__(self, total_epochs, batch_size, grad_accu_step, valid_step, log_step, save_step, data):
        
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.grad_accu_step = grad_accu_step
        self.valid_step = valid_step
        self.save_step = save_step
        self.log_step = log_step

        self.state = {
            'epoch': 0,
            'inner_step': 0, # num of data have been used
        }

        self.data = data

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

    def set_data(self, **config):

        self.dataset = {}

        for mode in self.data:
            for split in self.data[mode]:
                dataset_config = {**config['config'], **config['split'][split]}
                self.dataset[split] = get_dataset(config['wrapper'], dataset_config)
                print(len(self.dataset[split]))
        
        self._set_dataloader()

    def _set_dataloader(self):
        # mode in ['train'] ['valid']
        # split could be any thing
        self._dataloader = {}
        for mode in self.data:
            for split in self.data[mode]:
                self._dataloader[split] = DataLoader(
                    self.dataset[split],
                    batch_size = self.batch_size[mode],
                )

    def set_monitor(self, **config):
        self.monitor = get_monitor(**config)

    def get_data(self, split):
        return self._dataloader[split]

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
        pass

    def save(self):
        pass
