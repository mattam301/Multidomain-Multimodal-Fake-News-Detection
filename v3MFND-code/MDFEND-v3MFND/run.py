from utils.dataloader import w2v_data
import torch
import tqdm
import pickle
import logging
import os
import time
import json
from copy import deepcopy

from utils.utils import Averager
from utils.dataloader import bert_data
from models.mdfend import Trainer as MDFENDTrainer

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.fusion_source = config['fusion_source']
        self.fusion_type = config['fusion_type']
        self.use_cuda = config['use_cuda']
        self.with_emotion = config['with_emotion']
        self.cat_quantity = config['cat_quantity']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
## full domain
        self.train_path = self.root_path + 'train_vlsp2020_e_enhanced.pkl'
        self.val_path = self.root_path + 'val_vlsp2020_e_enhanced.pkl'
        self.test_path = self.root_path + 'test_vlsp2020_e_enhanced.pkl'
               
        self.category_dict_10 = {
            
            #  'DISASTERS': 0, 
            #  'FINANCE': 1,
            #  'POLITICS': 2, 
            #  'SOCIETY': 3,
            #  'HEALTH': 4
             
            "DISASTERS": 0,
            "EDUCATION": 1, 
            "ENTERTAINMENT": 2,
            "FINANCE": 3,
            "HEALTH": 4,
            "MILITARY": 5,
            "POLITICS": 6,
            "SCIENCE": 7,
            "SOCIETY": 8,
            "SPORTS": 9
        }

    def get_dataloader(self):
        if self.emb_type == 'bert':
            loader = bert_data(max_len = self.max_len, batch_size = self.batchsize, vocab_file = self.vocab_file,
                        category_dict_10 = self.category_dict_10, num_workers=self.num_workers)
            print(loader)
        elif self.emb_type == 'w2v':
            loader = w2v_data(max_len=self.max_len, vocab_file=self.vocab_file, emb_dim = self.emb_dim,
                    batch_size=self.batchsize, category_dict_10=self.category_dict_10, num_workers= self.num_workers)
            
        if self.cat_quantity == 10:
            train_loader = loader.load_data_10(self.train_path, True)
            val_loader = loader.load_data_10(self.val_path, False)
            test_loader = loader.load_data_10(self.test_path, False)
        if self.cat_quantity == 5:
            train_loader = loader.load_data_5(self.train_path, True)
            val_loader = loader.load_data_5(self.val_path, False)
            test_loader = loader.load_data_5(self.test_path, False)
        return train_loader, val_loader, test_loader
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        # print(self.mlp_dims)
        if self.model_name == 'mdfend':
            trainer = MDFENDTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, bert = self.bert, emb_type = self.emb_type,fusion_source = self.fusion_source, fusion_type = self.fusion_type,
                use_cuda = self.use_cuda,with_emotion=self.with_emotion, cat_quantity = self.cat_quantity, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict_10 = self.category_dict_10, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))  
        # print(torch.__version__)
        trainer.train()
