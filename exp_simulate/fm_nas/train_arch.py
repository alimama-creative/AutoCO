import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from model import NAS, NAS_TS, SNAS, DSNAS, SNAS_TS, DSNAS_TS
from ctr import TfsDataset, DirectDataset, DenseDataset
from torch.utils.data import DataLoader
import time
from utils import DataPrefetcher
from sklearn.metrics import roc_auc_score, log_loss
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_dataset(path, name, num=-1, flag=0):
    if name == 'tfs':
        return TfsDataset(path, num)
    if name == 'direct':
        return DirectDataset(path)
    if name == 'embedded':
        return DenseDataset(path, num, flag=flag)

class Train_Arch():
    def __init__(self,
                dim=20,
                weight_decay=0.001,
                data_size = -1,
                epoch=30,
                train_scale=0.4,
                valid_scale=0.4,
                learning_rate=0.001,
                batch_size=1024,
                device='cuda',
                params=None):
        self.device = torch.device(device)
        self.dim = dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.params = params
        self.model_nas = None
        self.data_size = data_size
        self.train_scale = train_scale
        self.valid_scale = valid_scale
        with open(params['feature_size'], 'rb') as f:           
            self.feature_size = pickle.load(f)
        print(self.feature_size)
        self.clock = [0 for _ in range(10)]
        self.run_first = 0
        self.last_arch = None
        self.model_path = self.params["model_path"]

    def set_dataloader(self, dataset=None, dataset_path=None, data_size=-1,train_scale=-1, valid_scale=-1, flag=0):
        if train_scale == -1:
            data_size = self.data_size
            train_scale = self.train_scale
            valid_scale = self.valid_scale
        self.log_data = dataset
        self.data_path = dataset_path
        if dataset == None:
            self.dataset = get_dataset(dataset_path, 'embedded', data_size, flag)
        else:
            self.dataset = get_dataset(dataset, 'direct')
        train_length = int(len(self.dataset) * train_scale)
        valid_length = int(len(self.dataset) * valid_scale)  
        test_length = len(self.dataset) - train_length - valid_length
        train_dataset, temp_dataset = torch.utils.data.random_split(
            self.dataset, (train_length, len(self.dataset) - train_length))
        valid_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, (valid_length, test_length))
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=32)
        # valid_size = int(len(valid_dataset)/len(train_data_loader))+1
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=32)
        self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=32)
        return self.train_data_loader, self.valid_data_loader, self.test_data_loader

    def get_model(self, model_nas, model_struct):
        if model_struct not in [0,1,2,3]:
            print("no model struct %s in model nas %s class!!!", model_struct, model_nas)
            exit()

        if model_nas == "nas":
            return NAS(self.feature_size, self.dim, self.params, m_type=model_struct)
        elif model_nas == "snas":
            return SNAS(self.feature_size, self.dim, self.params, m_type=model_struct)
        elif model_nas == "dsnas":
            return DSNAS(self.feature_size, self.dim, self.params, m_type=model_struct)
        elif model_nas == "nas_ts":
            return NAS_TS(self.feature_size, self.dim, self.params, m_type=model_struct)
        elif model_nas == "snas_ts":
            return SNAS_TS(self.feature_size, self.dim, self.params, m_type=model_struct)
        elif model_nas == "dsnas_ts":
            return DSNAS_TS(self.feature_size, self.dim, self.params, m_type=model_struct)
        else:
            print("no model named %s in nas train class!!!", model_nas)
            exit()

    def initial_model(self, model_nas, model_struct, optimizer='adam'):
        self.model_nas = model_nas
        self.model = self.get_model(model_nas, model_struct).to(self.device)
        # print(1)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.learning_rate, 
                                     weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.Adagrad(params=self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
        self.arch_optimizer = torch.optim.Adam(params=self.model.arch_parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
        if self.params["simulate_type"]=="click":
            self.criterion = torch.nn.BCELoss() #reduction="sum"
        else:
            # self.criterion = torch.nn.MSELoss()
            self.criterion = F.binary_cross_entropy
        
    def train_arch(self, train_type=0):
        self.model.train()
        losses = []
        prefetcher = DataPrefetcher(self.train_data_loader, self.device)
        total_y =[0,0]
        clock = [0 for _ in range(10)]
        clock[0] -= time.time()
        step = 0
        while 1:
            t=1
            clock[1] -= time.time()
            train_data = prefetcher.next()
            temperature = 2.5 * np.exp(-0.036 * step)
            if train_data == None:
                clock[1] += time.time()
                break
            (fields, target) = train_data
            fields, target = fields.to(self.device), target.to(self.device)
            clock[1] += time.time()
            if train_type == 0:
                if "dsnas" in self.model_nas:
                    others = self.optimizer
                elif "snas" in self.model_nas:
                    others = temperature
                else:
                    others = 0
                if "_ts" in self.model_nas:
                    loss = self.model.step_ts(fields, target, self.criterion, self.arch_optimizer, others)
                else:
                    loss = self.model.step(fields, target, self.criterion, self.arch_optimizer, others)
                losses.append(loss.cpu().detach().item())
            else:
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                if self.model_nas== "snas":
                    self.model.binarize(temperature)
                    y = self.model(fields, 1)
                    loss = self.criterion(y, target.float())
                elif "_ts" in self.model_nas:
                    self.model.binarize()
                    if train_type == 1:
                        k, y = self.model.forward_ts(fields, 1)
                    elif train_type == 2:
                        k, y = self.model.forward_ts(fields, 0)
                    # loss_t =  F.binary_cross_entropy(input=y,target=target,reduction='sum')
                    loss_t = self.criterion(y, target.float())*self.batch_size
                    alpha = 1/len(self.train_data_loader)
                    loss = alpha * k + loss_t
                    total_y[0]+=k
                    total_y[1]+=loss_t
                else:
                    self.model.binarize()
                    if train_type == 1:
                        y = self.model(fields, 1)
                    elif train_type == 2:
                        y = self.model(fields, 0)
                    loss = self.criterion(y, target.float())
                loss.backward()
                self.model.recover()
                self.optimizer.step()
                losses.append(loss.cpu().detach().item())

        clock[0] += time.time()
        print('cost:', [round(c, 5) for c in clock[0:7]])
        # print("model cost:", [round(c, 5) for c in self.model.cost])
        self.model.cost = [0 for _ in range(6)] 
        # g, gp = self.model.genotype()
        # print('genotype: %s' % g)
        # print('genotype_p: %s' % gp)
        if "nas_ts" in self.model_nas:
            print('kl distance ', total_y[0].item() / len(self.train_data_loader),'bce loss ', total_y[1].item() / len(self.train_data_loader))
        print('---- loss:', np.mean(losses))
        return np.mean(losses)

    def run(self):
        print("run!!!!!!")
        if self.params["trick"] == 0:
            arch = self.params["arch"]
            for j in range(len(self.model._arch_parameters["binary"])):
                self.model._arch_parameters["binary"].data[j, arch[j]] = 1
            best_loss = 1000000
            stop_cnt = 0
            self.model.genotype()
            for epoch_i in range(self.epoch):
                print(epoch_i, end=' ')
                loss = self.train_arch(train_type=2)
                if len(self.test_data_loader)>10:
                    self.test(self.test_data_loader)
                if best_loss - 0.000001 < loss:
                    stop_cnt += 1
                    if stop_cnt > 6:
                        break
                else:
                    best_loss = loss
                    self.save_model(self.model_path)
                    stop_cnt = 0
            self.load_model(self.model_path)
            if len(self.test_data_loader)>10:
                self.test(self.test_data_loader)
        elif self.params["trick"] == 1:
            for epoch_i in range(self.epoch):
                print(epoch_i, end=' ')
                self.train_arch()
                self.model.genotype()
                self.train_arch(train_type=2)
                if len(self.test_data_loader)> 10:
                    self.test(self.test_data_loader)
            best_loss = 1000000
            stop_cnt = 0
            for i in range(100):
                print(i, end=' ')
                loss = self.train_arch(train_type=2)
                if best_loss - 0.000001 < loss:
                    stop_cnt += 1
                    if stop_cnt > 6:
                        break
                else:
                    best_loss = loss
                    self.save_model(self.model_path)
                    stop_cnt = 0
            self.load_model(self.model_path)
        

    def test(self, data_loader):
        self.model.eval()
        self.model.binarize()
        targets, predicts, id_list = list(), list(), list()
        loss = 0
        with torch.no_grad():
            for fields, target in data_loader:
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields, 0)
                loss += F.binary_cross_entropy(input=y,target=target,reduction='mean')
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        self.model.recover()
        auc = roc_auc_score(targets, predicts)
        # print("loss: ", loss.item()/len(data_loader))
        print("auc: ", auc)#, "  g_auc:", cal_group_auc(targets, predicts, id_list)
        # print("bce:",  torch.nn.functional.binary_cross_entropy(input=predicts,target=target,reduction='mean'))
        return auc

    def result(self, candidates):
        self.model.eval()
        self.model.binarize()
        candidates = torch.from_numpy(candidates).to(self.device)
        if "_ts" in self.model_nas:
            _, ranking = self.model.forward_ts(candidates, 0, cal_kl=0)
        else:
            ranking = self.model(candidates, 0)
        self.model.recover()
        return ranking.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model_nas = self.params["model_nas"]
        self.model = torch.load(path)
    
    def print_w(self):
        self.model.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='raw_data/dataset/dataset_test.pkl')
    parser.add_argument('--feature_size', type=str, default='raw_data/data_online/feature_size.pkl')
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--learning', type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--dataset", type=int, default=-1)
    parser.add_argument("--times",type=int,default=1)
    parser.add_argument("--optimizer",type= str, default="adam")
    parser.add_argument('--model_struct', type=int, default=0)
    parser.add_argument('--model_nas', type=str, default="fm")
    args = parser.parse_args()

    params = {"device": args.device,
              "alpha": 0.0,
              "dense": [10,6,0],
              'feature_size': args.feature_size,
              "operator": "multiply",
              'trick': 2,
              "model_struct": args.model_struct,
              "model_nas": args.model_nas,
              "first_order": 1,
              "calcu_dense": 0,
              "f_len": 12,
              "early_fix_arch": False,
              "simulate_type":"click",
              'arch': [4,4,1,4,4,4,2,2,4,4,1,1,1,1,1]}  # [1,0,2,0,3,0,3,1,2,2,0,3,0,3,1,0,1,2,3,0,1,2]
    # [1,0,2,0,3,0,3,1,2,2,0,3,0,3,1][4,4,1,4,4,4,2,2,4,4,1,1,1,1,1]
    framework = Train_Arch(dim=int(args.dim),
                     weight_decay= float(args.decay),
                     data_size=int(args.dataset),
                     device=args.device,
                     train_scale=0.8,
                     valid_scale=0,
                     epoch=int(args.epoch),
                     params=params)
    
    result = []
    for i in range(args.times):
        search_start = time.time()
        framework.set_dataloader(dataset_path=args.dataset_path)
        framework.initial_model(args.model_nas, args.model_struct, "adam")
        # framework.params["trick"] = 0
        # framework.epoch = int(args.epoch)
        # framework.train_scale=0.4
        # framework.valid_scale=0.4
        # framework.set_dataloader(dataset_path=args.dataset_path)
        # # for j in range(len(framework.model._arch_parameters["binary"])):
        # #     framework.model._arch_parameters["binary"].data[j, np.random.randint(0,3)] = 1
        framework.params["arch"] = [1 for i in range(framework.model.multi_len)]
        framework.run()
        framework.model.genotype()
        # framework.test(framework.train_data_loader)
        # model_path = "../dataset/data_nas/raw_data/data_nas/rand_0.15_3/model/fm_nas.pt"
        # framework.save_model(model_path)
        print("---------------------")
        print("cost: ", time.time()-search_start)
        print("---------------------")
        # # framework.load_model(model_path)
        # arch = [framework.model._arch_parameters['binary'][i, :].argmax().item() 
        #         for i in range(len(framework.model._arch_parameters['binary']))]
        # # framework.params["arch"] = arch
        # framework.params["trick"] = 2
        # framework.epoch = 30
        # framework.train_scale=0.8
        # framework.valid_scale=0
        # framework.set_dataloader(dataset_path=args.dataset_path)
        
        # framework.run()
        result.append(framework.model.genotype())
        result.append(framework.test(framework.train_data_loader))
    print(result)
