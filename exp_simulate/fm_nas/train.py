import os
import time
import torch
import pickle

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

from ctr import TfsDataset, DirectDataset, DenseDataset
from model import OFM, OFM_TS
from utils import cal_group_auc, FTRL

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_dataset(path, name, num=-1, flag=0):
    if name == 'tfs':
        return TfsDataset(path, num)
    if name == 'direct':
        return DirectDataset(path)
    if name == 'embedded':
        return DenseDataset(path, num, flag=flag)


class Core(object):
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
        self.clock = [0 for _ in range(10)]

    def set_dataloader(self, dataset=None, dataset_path=None, data_size=-1,train_scale=-1, valid_scale=-1, flag=0):
        "split data into 3 part, train, test, valid"
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
        if model_struct not in [0,2]:
            print("no model struct %s in model nas %s class!!!", model_struct, model_nas)
            exit()

        if model_nas == "fm":
            model = OFM(self.feature_size, self.dim, self.params, m_type=model_struct)
            model.setotype(self.params["operator"])
            return model
        elif model_nas == "fm_ts":
            model = OFM_TS(self.feature_size, self.dim, self.params, m_type=model_struct)
            model.setotype(self.params["operator"])
            return model
        else:
            print("no model named %s in train class!!!", model_nas)
            exit()

    def initial_model(self, model_nas, model_struct, optimizer='adam'):
        self.model_nas = model_nas
        self.model = self.get_model(model_nas, model_struct).to(self.device)
        # print(1)
        if self.params["simulate_type"]=="click":
            self.criterion = torch.nn.BCELoss() #reduction="sum"
        else:
            self.criterion = F.binary_cross_entropy
            # self.criterion = torch.nn.MSELoss()
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer == 'ftrl':
            self.optimizer = FTRL(params=self.model.parameters(),
                                  alpha=1.0, beta=1.0, l1=0.001, l2=0.001)

    def train(self, data_loader, flag=0):
        self.model.train()
        total_loss = 0
        total_y =[0,0]
        M = len(data_loader)
        self.clock[0] -= time.time()
        for i, (fields, target) in enumerate(data_loader):
            self.clock[1] -= time.time()
            fields, target = fields.to(self.device), target.to(self.device)
            self.clock[1] += time.time()
            self.clock[2] -= time.time()
            if self.model_nas == "fm_ts":
                k, y = self.model.forward_ts(fields, 0)
                # loss_t = F.binary_cross_entropy(input=y,target=target,reduction='sum')
                loss_t = self.criterion(y, target.float())*self.batch_size
                alpha = 1/len(data_loader)
                loss = alpha * k + loss_t

                total_y[0]+=k
                total_y[1]+=loss_t
            else:
                y = self.model(fields, 0)
                loss = self.criterion(y, target.float())
            self.clock[2] += time.time()
            self.clock[3] -= time.time()
            self.model.zero_grad()
            loss.backward()
            self.clock[3] += time.time()
            self.clock[4] -= time.time()
            self.optimizer.step()
            self.clock[4] += time.time()
            total_loss += loss.item()
        self.clock[0] += time.time()
        if self.model_nas == "fm_ts":
            print('kl distance ', total_y[0].item() / len(data_loader),'bce loss ', total_y[1].item() / len(data_loader))
        print('------ loss:', total_loss / len(data_loader))
        
        print([round(c, 5) for c in self.clock[0:7]])
        # print(self.model.cost)
        self.model.cost = [0 for _ in range(6)]
        self.clock = [0 for _ in range(10)]
        return  total_loss / len(data_loader)

    def test(self, data_loader):
        self.model.eval()
        targets, predicts, id_list = list(), list(), list()
        loss = 0
        with torch.no_grad():
            for fields, target in data_loader:
                fields, target = fields.to(self.device), target.to(self.device)
                if self.model_nas == "fm_ts":
                    _,y = self.model.forward_ts(fields, 0, cal_kl=1)
                else:
                    y = self.model(fields, 0)
                loss += torch.nn.functional.binary_cross_entropy(input=y,target=target,reduction='sum')
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                for i in fields.tolist():
                    id_list.append(",".join([str(s) for s in i[:68]]))
        # print("loss: ", loss.item()/len(data_loader))
        auc = roc_auc_score(targets, predicts)
        print("auc: ", auc)#, "  g_auc:", cal_group_auc(targets, predicts, id_list)
        # print("bce:",  torch.nn.functional.binary_cross_entropy(input=predicts,target=target,reduction='mean'))
        return auc

    def run(self, flag=0):
        epoch = self.epoch
        start = time.time()
        stop_cnt = 0
        best_loss = 10000000
        for i in range(epoch):
            print(i, end=" ")
            loss = self.train(self.train_data_loader)
            if best_loss - 0.000001 < loss:
                stop_cnt += 1
                if stop_cnt > 6:
                    break
            else:
                best_loss = loss
                stop_cnt = 0
        print(len(self.test_data_loader))
        if len(self.test_data_loader)> 10:
            self.test(self.test_data_loader)
            # if i%10==0 or epoch-i<4:
            #     self.print_w()
            # if i%10 == 9:
            #     print('epoch:', i + 1)
        print("cost time: ", time.time()-start)

    def result(self, candidates):
        self.model.eval()
        candidates = torch.from_numpy(candidates).to(self.device)
        if self.model_nas == "fm_ts":
            _, ranking = self.model.forward_ts(candidates, 0, cal_kl=0)
        else:
            ranking = self.model(candidates, 0)
        return ranking.detach().cpu().numpy()


    def print_w(self):
        self.model.print()

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='raw_data/dataset/dataset_1400.pkl')
    parser.add_argument('--feature_size', type=str, default='raw_data/data_1400w/feature_size_id.pkl')
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--learning', type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--dataset", type=int, default=-1)
    parser.add_argument("--times",type=int,default=1)
    parser.add_argument("--optimizer",type= str, default="adam")
    parser.add_argument("--model", type=str, default="fm")
    parser.add_argument("--oper", type=str, default="multiply")
    args = parser.parse_args()

    params = {"device": args.device,
              "alpha": 0.0,
              "dense": [0,5,18],
              'feature_size': args.feature_size,
              "operator": args.oper,
              'trick': 2,
              "first_order": 1,
              "calcu_dense": 0,
              'arch':[1,0,2,0,3,0,3,1,2,2,0,3,0,3,1] } # [1,0,2,0,3,0,3,1,2,2,0,3,0,3,1,0,1,2,3,0,1,2]

    framework = Core(dim=int(args.dim),
                     weight_decay= float(args.decay),
                     data_size=int(args.dataset),
                     device=args.device,
                     split_param=0.8,
                     epoch=int(args.epoch),
                     params=params)
    framework.set_dataloader(dataset_path=args.dataset_path)

    for i in range(args.times):
        search_start = time.time()
        framework.initial_model(args.model)
        # for j in range(len(framework.model._arch_parameters["binary"])):
        #     framework.model._arch_parameters["binary"].data[j, np.random.randint(0,3)] = 1
        framework.run()
        # framework.model.genotype()
        framework.test(framework.train_data_loader)
        
        print("---------------------")
        print("cost: ", time.time()-search_start)
        print("---------------------")
        # framework.load_model(model_path)
        framework.print_w()
        # with open("../dataset/data_nas/raw_data/data_nas/model_param_0.15.pkl", 'rb') as f:
        #     param = torch.load(f)
        # cnt = 0
        # for name, parameter in framework.model.named_parameters():
        #     parameter[:] = param[cnt]
        #     cnt += 1
        # framework.test(framework.train_data_loader)
