import os
import time
import pickle
import random
import numpy as np
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .summary_board import SummaryBoard


class EpochBasedTrainer:
    def __init__(self, cfg):
        self.snapshot_prefix = cfg.runname
        self.max_epoch = cfg.optim.max_epoch
        self.log_steps = cfg.log_steps
        self.save_steps = cfg.save_steps
        self.log_dir = cfg.log_dir + '/'
        self.snapshot_dir = cfg.snapshot_dir + '/'

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #torch.autograd.set_detect_anomaly(True)

        if not os.path.exists(cfg.snapshot_dir):
            os.makedirs(cfg.snapshot_dir)
        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)

        self.log_file:str = cfg.log_dir + '/' + cfg.runname + '_{}.pkl'
        self.log_file = self.log_file.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)

        # deep learning entities
        self.model: Optional[nn.Module] = None
        self.evaluator: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.StepLR] = None
        self.loss_func: Optional[nn.Module] = None
        self.clip_grad = cfg.optim.clip_grad_norm

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.epoch = 0
        self.iteration = 0
    

    @classmethod
    def release_cuda(clf, x):
        if isinstance(x, list):
            x = [clf.release_cuda(item) for item in x]
        elif isinstance(x, tuple):
            x = (clf.release_cuda(item) for item in x)
        elif isinstance(x, dict):
            x = {key: clf.release_cuda(value) for key, value in x.items()}
        elif isinstance(x, torch.Tensor):
            if x.numel() == 1:
                x = x.item()
            else:
                x = x.detach().cpu().numpy()
        return x
    
    @classmethod
    def to_cuda(clf, x):
        if isinstance(x, list):
            x = [clf.to_cuda(item) for item in x]
        elif isinstance(x, tuple):
            x = (clf.to_cuda(item) for item in x)
        elif isinstance(x, dict):
            x = {key: clf.to_cuda(value) for key, value in x.items()}
        elif isinstance(x, torch.Tensor):
            x = x.cuda()
        return x
    
    def save_model(self, filename):
        filename = self.snapshot_prefix + filename + ".pth"
        torch.save(self.model.state_dict(), self.snapshot_dir + filename)
        print('Model saved to "{}"'.format(filename))
    
    def save_snapshot(self, filename):
        state_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        filename = self.snapshot_prefix + filename + '.pth.tar'
        torch.save(state_dict, self.snapshot_dir + filename)
        print('Snapshot saved to "{}"'.format(filename))
    
    def load_snapshot(self, filename):
        print('Loading from "{}".'.format(filename + '.pth.tar'))
        state_dict = torch.load(self.snapshot_dir + filename + '.pth.tar')
        self.model.load_state_dict(state_dict['model'], strict=False)
        print('Model has been loaded.')

        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
        if 'scheduler' in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            except ValueError:
                pass
    

    def set_train_mode(self):
        self.model.train()
        if self.evaluator is not None:
            self.evaluator.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.model.eval()
        if self.evaluator is not None:
            self.evaluator.eval()
        torch.set_grad_enabled(False)
    
    
    def step(self, data_dict) -> Dict[str,torch.Tensor]:
        if isinstance(data_dict, tuple) or isinstance(data_dict, list):
            output_dict = self.model(*data_dict)
        else: output_dict = self.model(data_dict)
        loss_dict: Dict = self.loss_func(output_dict, data_dict)
        if self.evaluator is not None:
            with torch.no_grad():
                result_dict = self.evaluator(output_dict, data_dict)
            loss_dict.update(result_dict)
        return loss_dict
    
    def train_epoch(self):
        self.optimizer.zero_grad()
        steps = len(self.train_loader)
        for iteration, data_dict in enumerate(self.train_loader):
            self.iteration += 1
            data_dict = self.to_cuda(data_dict)
            result_dict = self.step(data_dict)
            result_dict['loss'].backward()

            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            result_dict = self.release_cuda(result_dict)
            self.summary_board.update_from_dict(result_dict)
            #torch.cuda.empty_cache()

            print("Epoch %d [%d/%d]"%(self.epoch, iteration+1, steps), end=' ')
            for key, value in result_dict.items():
                print(key, "%.4f"%float(value), end='; ')
            print()

            if (iteration + 1) % self.log_steps == 0:
                logs = dict()
                for k,v in self.summary_board.meter_dict.items():
                    logs[k] = np.array(v._records)
                print("Logging into ", self.log_file)
                flog = open(self.log_file, 'wb')
                flog.write(pickle.dumps(logs))
                flog.close()
            
            if self.save_steps > 0 and (iteration + 1) % self.save_steps == 0:
                self.save_snapshot("-epoch-%02d-%d"%(self.epoch, iteration + 1))
                if self.scheduler is not None: self.scheduler.step()
        
        if self.scheduler is not None:
            self.scheduler.step()


    def validate_epoch(self):
        self.set_eval_mode()
        summary_board = SummaryBoard(adaptive=True)
        print("---------Start validation---------")
        torch.cuda.synchronize()
        start = time.time()

        for iteration, data_dict in enumerate(self.val_loader):
            data_dict = self.to_cuda(data_dict)
            result_dict = self.step(data_dict)
            result_dict = self.release_cuda(result_dict)
            
            summary_board.update_from_dict(result_dict)
            print("[%d/%d]"%(iteration, len(self.val_loader)), end=' ')
            for key, value in result_dict.items():
                print(key, "%.4f"%float(value), end='; ')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print('%.4fs'%(time.time() - start))
        
        self.set_train_mode()
        summary = summary_board.summary()
        summary_dict = {"val_" + k : v for k,v in summary.items()}
        self.summary_board.update_from_dict(summary_dict)
        
        print("Validate Epoch %02d:"%self.epoch, end=' ')
        for key, value in summary_dict.items():
            print(key, "%.4f"%float(value), end='; ')
        print()

        logs = dict()
        for k,v in self.summary_board.meter_dict.items():
            logs[k] = np.array(v._records)
        flog = open(self.log_file, 'wb')
        flog.write(pickle.dumps(logs))
        flog.close()

        return summary_board


    def fit(self, resume_epoch=0, resume_log=None):
        assert self.train_loader is not None
        if resume_log is not None and resume_epoch > 0:
            self.load_snapshot(self.snapshot_prefix + "-epoch-%02d"%resume_epoch)
            print('Continue training from epoch %02d.'%(self.epoch + 1))
            f = open(self.log_dir + resume_log, 'rb')
            log: Dict[str, np.ndarray] = pickle.load(f)
            f.close(); data = dict()
            for k, v in log.items():
                print(k, v.shape)
                if k[:3] == 'val': data[k] = v[:self.epoch].tolist()
                else: data[k] = v[:self.iteration+1].tolist()
            self.summary_board.update_from_dict(data)
        self.set_train_mode()

        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            if self.val_loader is not None:
                self.validate_epoch()
            self.save_snapshot("-epoch-%02d"%self.epoch)