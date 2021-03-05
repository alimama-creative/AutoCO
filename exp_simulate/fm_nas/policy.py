import time
import torch
import pickle
import numpy as np
import pandas as pd

from torch.utils.data import WeightedRandomSampler

from train import Core
from train_arch import Train_Arch
from model import Linucb


def get_creative(params):
    "return global creative list, and item-creatives dict"
    with open(params['creative_list'], 'rb') as f:
        c_list = pickle.load(f)
    c_list = np.array(c_list, dtype=np.float32)
    with open(params['item_candidates'], 'rb') as f:
        item_candi = pickle.load(f)
    with open(params["feature_list"], 'rb') as f:
        f_list = pickle.load(f)
    return f_list, c_list, item_candi


class Base(object):
    def __init__(self, params):
        self.f_list, self.c_list, self.item_candi = get_creative(params)
        self.name = 'Base'
        self.params = params
        self.c_len = len(self.c_list[0])
        self.f_len = len(self.f_list[0])
        self.t_len = self.f_len+self.c_len
        self.params["c_len"] = self.c_len
        self.params["f_len"] = self.f_len
        self.log_file = None
        self.log_data = []
        self.batch_data = []
        self.fea_index = {}
        self.c_index = {}
        self.cnt = 0
        cnt= 0
        for cre in self.c_list:
            self.c_index[','.join(str(x) for x in cre)] = cnt
            cnt += 1
        self.clock = [0 for _ in range(10)]

    def update(self, lines, size=-1):
        self.log_data.extend(lines)
        self.batch_data = lines
        if size!=-1 and len(self.log_data)>size:
            self.log_data = self.log_data[-size:]
        return 0

    def update_in_log(self):
        """write log into file"""
        if len(self.log_data)==0:
            return
        df = pd.DataFrame(self.log_data)
        df.to_pickle(self.log_file)

    def get_recommend_info(self, features_list, flag=0):
        """
            Return several array for calculate
            feas: distinct feature list which have not been displayed
            offset: record the beginning of feature creatives
            candidates: creatives of all features waiting for ranking
        """
        feas = []
        num = 0
        for features in features_list:
            ind = features[0]
            if ind not in self.fea_index.keys():
                self.fea_index[ind] = self.cnt
                self.cnt += 1
                feas.append(features) 
                num += len(self.item_candi[ind])
        cnt = 0
        f_len = len(features_list[0][1:])
        # print(len(self.c_list[0]))
        if flag == 0:
            leng = f_len+len(self.c_list[0])
        else:
            leng = f_len+len(self.c_list[0])+1
        candidates = np.zeros((num, leng),dtype=np.float32)
        offset = [0]
        last = 0
        for features in feas:
            t = np.zeros(leng)
            if flag == 0:
                t[0:f_len] = np.array(features[1:], dtype=np.float32)
                for c_feature in self.item_candi[features[0]]:
                    t[f_len:] = self.c_list[c_feature]
                    candidates[cnt] = t
                    cnt+=1
            else:
                t[1:1+f_len] = np.array(features[1:], dtype=np.float32)
                for c_feature in self.item_candi[features[0]]:
                    t[0] = np.float32(c_feature)
                    t[1+f_len:] = self.c_list[c_feature]
                    candidates[cnt] = t
                    cnt+=1
            last = last+len(self.item_candi[features[0]])
            offset.append(last)
        return feas, offset, candidates


class Random(Base):
    def __init__(self, params):
        super(Random, self).__init__(params)
        self.name = 'Random'
        self.log_file = self.params["random_file"]

    def update(self, lines, ind=None):
        super(Random, self).update(lines)
        return 0
 
    def recommend_batch(self, features_list):
        res = []
        for features in features_list:
            leng = len(self.item_candi[features[0]])
            res.append(np.random.randint(0,leng))
        return res


class Greedy(Base):
    def __init__(self, params):
        super(Greedy, self).__init__(params)
        self.name = 'Greedy'

    def update(self, lines, ind):
        return 0

    def recommend_batch(self, features_list, item_ctr):
        res = []
        for features in features_list:
            res.append(item_ctr[features[0]].index(max(item_ctr[features[0]])))
        return res


class FmEGreedy(Base):
    def __init__(self, params):
        super(FmEGreedy, self).__init__(params)
        self.log_file = 'data/log/fm_log.pkl'
        self.name = 'FmEGreedy'
        self.fea_index = {}
        self.res = []
        self.flag = 0
        self.update_cnt = 0
        self.greedy_flag = 0
        self.epsilon = self.params["alpha"]
        self.model_nas = self.params["model_nas"]
        self.model_struct = self.params["model_struct"]

        # intial model if nas model, use Train_Arch() , else use Core()
        if self.model_nas not in ["fm", "fm_ts"]:
            self.framework = Train_Arch(dim=self.params["dim"],epoch=self.params["epoch"],weight_decay=self.params["decay"],data_size=self.params["data_size"], train_scale=1, valid_scale=0, device=self.params["device"], params=self.params)
        else:
            self.framework = Core(dim=self.params["dim"],epoch=self.params["epoch"],weight_decay=self.params["decay"],data_size=self.params["data_size"],train_scale= 1, valid_scale=0, device=self.params["device"], params=self.params)

        # get test dateset to calculate auc
        if self.params["auc_record"] == 1:
            _, _, self.test_data = self.framework.set_dataloader(dataset_path=self.params["random_file"], data_size=100000,train_scale=0,valid_scale=0)


    def update(self, lines, ind=None):
        self.update_cnt += len(lines)
        if self.name == "FmTS":
            self.framework.epoch = self.framework.params["epoch"]-int((len(self.log_data)/len(lines))*3.5)
            print("epoch:", self.framework.epoch)
        
        # update == 0, the struct of model will update frequently
        if self.params["update"] == 0:
            super(FmEGreedy, self).update(lines)
            self.framework.set_dataloader(dataset=self.log_data, dataset_path=self.log_file)
            self.framework.initial_model(self.model_nas, self.model_struct, optimizer=self.params["optimizer"])
            self.framework.run()

        # update == 1, the struct of model will update several batch a time
        elif self.params["update"] == 1:
            super(FmEGreedy, self).update(lines)
            if self.update_cnt > 49999 or self.flag ==0:
                self.framework.params["trick"]=1
                self.framework.epoch = self.framework.params["epoch"]
                self.framework.set_dataloader(dataset=self.log_data, dataset_path=self.log_file)
                self.framework.initial_model(self.model_nas, self.model_struct, optimizer=self.params["optimizer"])
                self.framework.run()
                self.framework.params["arch"] = [self.framework.model._arch_parameters['binary'][i, :].argmax().item() for i in range(len(self.framework.model._arch_parameters['binary']))]
                self.flag, self.update_cnt = 1, 0
            else:
                self.framework.epoch = 200
                self.framework.params["trick"]=0
                if self.name == "FmTS":
                    self.framework.epoch = 180-int(len(self.log_data)/len(lines))*6
                self.framework.set_dataloader(dataset=self.log_data, dataset_path=self.log_file)
                self.framework.initial_model(self.model_nas, self.model_struct, optimizer=self.params["optimizer"])
                # if self.flag == 0:
                #     self.framework.params["arch"] = [1 for i in range(len(self.framework.model._arch_parameters['binary']))]
                self.framework.run()

        self.fea_index = {}
        self.res = []
        self.cnt = 0
        if self.params["auc_record"] == 1:
            return(self.framework.test(self.test_data))
        else:
            return 0

    def recommend_batch(self, features_list):
        feas, offset, candidates = super().get_recommend_info(features_list)
        if len(candidates) != 0:
            rank = self.framework.result(candidates)
            for i in range(len(feas)):
                k = np.argmax(rank[offset[i]:offset[i+1]])
                self.res.append(k)
        final = []
        for features in features_list:
            if np.random.rand() > self.epsilon:
                final.append(self.res[self.fea_index[features[0]]])
            else:
                leng = len(self.item_candi[features[0]])
                final.append(np.random.randint(0,leng))
        return final


class FmThompson(FmEGreedy):
    def __init__(self, params):
        if "_ts" not in params["model_nas"]:
            params["model_nas"] = params["model_nas"]+"_ts"
        super(FmThompson, self).__init__(params)
        self.log_file = 'data/log/fmTS_log.pkl'
        self.name = 'FmTS'
        self.epsilon = 0

    def update(self, lines, ind=None):
        return super(FmThompson, self).update(lines)

    def recommend_batch(self, features):
        return super(FmThompson, self).recommend_batch(features)


class FmGreedy(FmEGreedy):
    def update(self, lines, ind):
        if self.greedy_flag == 1:
            return 0
        self.log_file = self.params["random_file"]
        self.greedy_flag = 1
        self.framework.set_dataloader(dataset=None, dataset_path=self.log_file)
        self.framework.initial_model(self.model_nas, self.model_struct, optimizer=self.params["optimizer"])
        self.framework.run()
        
        _, _, self.test_data = self.framework.set_dataloader(dataset_path=self.params["random_file"], data_size=100000,train_scale=0,valid_scale=0, flag=1)
        if self.params["auc_record"] == 1:
            return(self.framework.test(self.test_data))
        else:
            return 0

    
class LinUCB(Base):
    def __init__(self, params):
        super(LinUCB, self).__init__(params)
        self.log_file = 'data/log/linucb_log.pkl'
        self.name = 'Linucb'
        self.device = params['device']
        self.r_index = {}
        self.c_num = len(self.c_list)
        self.leng = self.t_len
        self.alpha = self.params["alpha"]
        self.cnt = 0
        self.t = 0

        self.Aa = np.zeros((self.c_num, self.leng, self.leng))
        self.Aa_inv = np.zeros((self.c_num, self.leng, self.leng))
        self.ba = np.zeros((self.c_num, self.leng, 1))
        self.theta = np.zeros((self.c_num, self.leng, 1))
        for i in range(self.c_num):
            self.Aa[i] = np.identity(self.leng)
            self.Aa_inv[i] = np.identity(self.leng)
            self.ba[i] = np.zeros((self.leng, 1))
            self.theta[i] = np.zeros((self.leng, 1))
        
        self.model = Linucb(self.leng, self.c_num, self.device)
        
    def update(self, lines, ind=None):
        # super(LinUCB, self).update(lines)
        used = np.zeros(self.c_num)
        # print(self.c_index)
        for line in lines:
            curr = self.c_index[','.join(str(float(x)) for x in line[-1-self.c_len:-1])]
            used[curr] = 1
            x = np.array(line[0:-1]).reshape((self.leng, 1))
            # print(x)
            reward = line[-1]
            t = np.outer(x, x)
            self.Aa[curr] += t
            self.ba[curr] += reward * x
            self.Aa_inv[curr] = self.Aa_inv[curr] - np.matmul(self.Aa_inv[curr].dot(x),x.T.dot(self.Aa_inv[curr]))/(1 + np.matmul(x.T, self.Aa_inv[curr].dot(x)))
        for curr in range(self.c_num):
            if used[curr] == 1:
                # self.Aa_inv[curr] = np.linalg.inv(self.Aa[curr])
                self.theta[curr] = self.Aa_inv[curr].dot(self.ba[curr])
        for i, (_, param) in enumerate(self.model.named_parameters()):
            if i ==0:
                param.data = torch.from_numpy(self.theta.reshape(self.c_num,self.leng).astype(np.float32)).to(self.device)
            if i == 1:
                param.data = torch.from_numpy(self.Aa_inv.astype(np.float32)).to(self.device)  
        self.r_index = {}
        self.fea_index = {}
        return 0
    
    def recommend_batch(self, features_list):
        feas, offset, candidates = super().get_recommend_info(features_list, 1)
        rank = self.model(torch.from_numpy(candidates).to(self.device))
        rank = rank.detach().cpu().numpy().reshape(len(rank))
        for i in range(len(feas)):
            k = np.argmax(rank[offset[i]:offset[i+1]])
            self.r_index[feas[i][0]] = k
        final = []
        for features in features_list:
            final.append(self.r_index[features[0]])
        return final


class TS(LinUCB):
    def __init__(self, params):
        super(TS, self).__init__(params)
        self.log_file = 'data/log/ts_log.pkl'
        self.name = 'ts'
        self.mean = None
        self.std = None
        self.alpha = self.params["alpha"]

    def update(self, lines, ind=None):
        for line in lines:
            curr = self.c_index[','.join(str(float(x)) for x in line[-1-self.c_len:-1])]
            x = np.array(line[0:-1]).reshape((self.leng, 1))
            reward = line[-1]
            t = np.outer(x, x)
            self.Aa[curr] += t
            self.ba[curr] += reward * x
            self.Aa_inv[curr] = self.Aa_inv[curr] - np.matmul(self.Aa_inv[curr].dot(x),x.T.dot(self.Aa_inv[curr]))/(1 + x.T.dot(self.Aa_inv[curr]).dot(x))
            t =self.Aa_inv[curr].dot(self.ba[curr])
            self.theta[curr] = t
        self.r_index = {}
        self.fea_index = {}
        self.mean = torch.from_numpy(self.theta).reshape(self.c_num, self.leng)
        temp = np.array([np.diag(a) for a in self.Aa_inv])
        self.std = torch.from_numpy(temp.reshape((self.c_num, self.leng)))

    def recommend_batch(self, features_list):
        theta = torch.normal(self.mean, self.alpha*self.std).numpy().reshape((self.c_num, self.leng))
        feas, offset, candidates = super().get_recommend_info(features_list)
        for i in range(len(feas)):
            ind = feas[i][0]
            c_num = len(self.item_candi[ind])
            res = np.zeros(c_num)
            for j in range(c_num):
                res[j] = theta[self.item_candi[ind][j]].T.dot(candidates[offset[i]+j])
            k = np.argmax(res)
            self.r_index[ind] = k
        final = []
        for features in features_list:
            final.append(self.r_index[features[0]])
        return final