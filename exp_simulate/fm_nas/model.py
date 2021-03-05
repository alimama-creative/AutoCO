import torch
import torch.nn as nn
import torch.distributions.normal as normal
import torch.nn.functional as F
import math
import time
import numpy as np
import random
from torch.autograd import Variable
PRIMITIVES = ['concat', 'multiply', 'max', 'min', 'plus']
# PRIMITIVES = ['zero', 'max', 'min', 'multiply', 'plus', 'minus']
OPS = {
    'zero': lambda p,q: torch.zeros_like(p).sum(2),
    'plus': lambda p,q: (p + q).sum(2),
    'minus': lambda p,q: (p - q).sum(2),
    'multiply': lambda p, q: (p * q).sum(2),
	'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0].sum(2),
	'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0].sum(2),
    'concat': lambda p, q:  torch.cat([p, q], dim=-1).sum(2)
}
OPS_V = {
    'zero': lambda p,q: torch.zeros_like(p),
    'plus': lambda p,q: (p + q),
    'minus': lambda p,q: (p - q),
    'multiply': lambda p, q: (p * q),
	'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0],
	'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0],
    'concat': lambda p, q:  torch.cat([p, q], dim=-1)
}


def MixedBinary(embedding_p, embedding_q, weights, flag, dim=1, FC=None, o_type=0, others=0):
    # print(weights)
    # embedding_p = MLP[0](embedding_p.view(-1,1)).view(embedding_p.size())
    # embedding_q = MLP[1](embedding_q.view(-1,1)).view(embedding_q.size())
    if flag == 0:
        max_ind = weights.argmax().item()
        if o_type==0:
            t = OPS[PRIMITIVES[max_ind]](embedding_p, embedding_q)
        elif o_type==1:
            begin, end = max_ind*dim, max_ind*dim+dim
            t = OPS[PRIMITIVES[max_ind]](embedding_p[:,:,begin:end], embedding_q[:,:,begin:end])
        elif o_type==2:
            t = (FC[max_ind](OPS_V[PRIMITIVES[max_ind]](embedding_p, embedding_q))).squeeze(1)
        elif o_type==3:
            begin, end = max_ind*dim, max_ind*dim+dim
            t = (FC[max_ind](OPS_V[PRIMITIVES[max_ind]](embedding_p[:,:,begin:end], embedding_q[:,:,begin:end]))).squeeze(1)
        if others == 1:
            t = t*weights[max_ind]
    else:
        if o_type==0:
            t = torch.sum(torch.stack([w * (OPS[primitive](embedding_p, embedding_q)) for w,primitive in zip(weights, PRIMITIVES)]), 0)
        elif o_type==1:
            t = 0
            for i in range(len(PRIMITIVES)):
                begin, end = i*dim, i*dim+dim
                t += weights[i] * (OPS[PRIMITIVES[i]](embedding_p[:,:,begin:end], embedding_q[:,:,begin:end]))
        elif o_type==2:
            t = 0
            for i in range(len(PRIMITIVES)):
                t += weights[i] * (FC[i](OPS_V[PRIMITIVES[i]](embedding_p, embedding_q))).squeeze(1)
        elif o_type==3:
            t = 0
            for i in range(len(PRIMITIVES)):
                begin, end = i*dim, i*dim+dim
                t += weights[i] * (FC[i](OPS_V[PRIMITIVES[i]](embedding_p[:,:,begin:end], embedding_q[:,:,begin:end]))).squeeze(1)

    return t


class OFM(nn.Module):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__()
        print(feature_size)
        self.name = "OFM"
        self.field_size = len(feature_size)
        print(feature_size)
        self.feature_sizes = feature_size
        self.dim = k
        self.device = params["device"]
        self.dense = params["dense"]
        self.first_order = params["first_order"]
        self.type = m_type
        self.f_len = params["f_len"]
        self.alpha = params["alpha"]
        # self.nas_flag = nas_flag
        # init bias
        self.bias = nn.Parameter(torch.normal(torch.ones(1), 1),requires_grad=True)
        self.params = params
        # init first order
        if self.first_order == 1:
            fm_first_order_Linears = nn.ModuleList(
                    [nn.Linear(feature_size, 1, bias=False) for feature_size in self.feature_sizes[:self.dense[0]]])
            fm_first_order_embeddings = nn.ModuleList(
                    [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes[self.dense[0]:self.dense[0]+self.dense[1]]])
            fm_first_order_multi = nn.Embedding(self.dense[2], 1) 
            self.fm_first_order_models = fm_first_order_Linears.extend(fm_first_order_embeddings).append(fm_first_order_multi)
            # self.bias = 0

        # init second order
        self._FC = None
        if self.type == 0:
            leng = self.dim
        elif self.type == 1:
            leng = self.dim*len(PRIMITIVES)
        elif self.type == 2:
            leng = self.dim
            # leng = self.dim
            self._FC = nn.ModuleList()
            for primitive in PRIMITIVES:
                if primitive == "concat":
                    self._FC.append(nn.Linear(2*self.dim, 1, bias=False))
                else:
                    self._FC.append(nn.Linear(self.dim, 1, bias=False))
        elif self.type == 3:
            leng = self.dim*len(PRIMITIVES)
            # leng = self.dim
            self._FC = nn.ModuleList()
            for primitive in PRIMITIVES:
                if primitive == "concat":
                    self._FC.append(nn.Linear(2*self.dim, 1, bias=False))
                else:
                    self._FC.append(nn.Linear(self.dim, 1, bias=False))

        fm_second_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, leng, bias=False) for feature_size in self.feature_sizes[:self.dense[0]]])
        fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, leng) for feature_size in self.feature_sizes[self.dense[0]:self.dense[0]+self.dense[1]]])
        fm_second_order_multi = nn.Embedding(self.dense[2], leng) 
        self.fm_second_order_models = fm_second_order_Linears.extend(fm_second_order_embeddings).append(fm_second_order_multi)

         # calcu the num of operation
        if self.dense[2] != 0:
            self.multi_len = int(self.dense[1]*(self.dense[1]+1)/2)+1
        else:
            self.multi_len = int(self.dense[1]*(self.dense[1]-1)/2)

        # init arch parameters
        self.inter_arr = [i for i in range(self.multi_len)]
        self._arch_parameters = {}
        self._arch_parameters['binary'] = Variable(torch.ones((self.multi_len, len(PRIMITIVES)), dtype=torch.float, device=params['device']) / 2, requires_grad=True)
        self._arch_parameters['binary'].data.add_(
            torch.randn_like(self._arch_parameters['binary'])*1e-3)
        

        self.cost = [0 for _ in range(6)]

    def arch_parameters(self):
        return [self._arch_parameters['binary']]

    def set_rand(self, num):
        self.inter_arr = random.sample(self.inter_arr, num)

    def forward(self, x, flag, weights=None):
        if weights == None:
            weights = self._arch_parameters['binary']
        X = x.reshape(x.shape[0], x.shape[1], 1)
        self.cost[0] -= time.time()

        # calcu first order
        out_first_order = 0
        if self.first_order == 1:
            for i, emb in enumerate(self.fm_first_order_models):
                if i < self.dense[0]:
                    Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.float)
                    out_first_order += torch.sum(emb(Xi_tem).unsqueeze(1), 1)
                elif i < self.dense[0]+self.dense[1]:
                    self.cost[1] -= time.time()
                    Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.long)
                    # print(i, Xi_tem, emb, self.feature_sizes)
                    out_first_order += torch.sum(emb(Xi_tem), 1)
                    self.cost[1] += time.time()
                else:
                    self.cost[2] -= time.time()
                    for j in range(self.dense[2]):
                        Xi_tem = X[:, i+j, :].to(device=self.device, dtype=torch.long)
                        out_first_order += Xi_tem*emb(torch.Tensor([j]).to(device=self.device, dtype=torch.long))
                    self.cost[2] += time.time()

        # record second order embedding
        X_vector = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i < self.dense[0]:
                if self.params["calcu_dense"]==0:
                    continue
                else:
                    Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.float)
                    X_vector.append(emb(Xi_tem).unsqueeze(1))
            elif i < self.dense[0]+self.dense[1]:
                self.cost[3] -= time.time()
                Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.long)
                X_vector.append(emb(Xi_tem))
                # print(X_vector[-1].shape)
                self.cost[3] += time.time()
            else:    
                self.cost[4] -= time.time()
                for j in range(self.dense[2]):
                    Xi_tem = X[:, i+j, :].to(device=self.device, dtype=torch.long)
                    X_vector.append((Xi_tem*emb(torch.Tensor([j]).to(device=self.device, dtype=torch.long))).unsqueeze(1))
                    # print(X_vector[-1].shape)
                self.cost[4] += time.time()

        # calcu second order
        out_second_order = 0
        self.cost[5] -= time.time()
        cnt = 0
        multi_hot_len = 0
        if self.dense[2] != 0:
            multi_hot_len = 1
        for i in range(len(X_vector)):
            for j in range(i):
                if i < len(self.feature_sizes)-multi_hot_len:
                    tmp = cnt
                elif j < len(self.feature_sizes)-multi_hot_len:
                    tmp = self.multi_len-len(self.feature_sizes)+j
                else:
                    tmp = self.multi_len-1
                cnt += 1
                if tmp not in self.inter_arr:
                    continue
                # if self.name != "SNAS":
                #     out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp,:], flag, dim=self.dim, FC=self._FC, o_type=self.type)
                # else:
                #     out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp], flag, dim=self.dim, FC=self._FC, o_type=self.type)
                if self.name == "SNAS":
                    out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp], flag, dim=self.dim, FC=self._FC, o_type=self.type)
                elif self.name == "DSNAS":
                    out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp, :], flag, dim=self.dim, FC=self._FC, o_type=self.type, others=1)
                else:
                    out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp, :], flag, dim=self.dim, FC=self._FC, o_type=self.type)
        # print(out_second_order)
        self.cost[5] += time.time()
        self.cost[0] += time.time()         
        out = out_second_order+out_first_order+self.bias

        return torch.sigmoid(out.squeeze(1))

    def genotype(self):
        genotype = [PRIMITIVES[self._arch_parameters['binary'][i, :].argmax().item()] 
                     for i in range(self.multi_len)]
        for i in range(self.multi_len):
            if i not in self.inter_arr:
                genotype[i] = "None"
        print(genotype)
        return genotype
        # return genotype, genotype_p.cpu().detach()

    def setotype(self, ops):
        self.ops = ops
        leng = len(self._arch_parameters['binary'][0,:])
        for i in range(self.multi_len):
            for j in range(leng):
                if  PRIMITIVES[j] == self.ops:
                    self._arch_parameters['binary'].data[i,j] = 1.0
                else:
                    self._arch_parameters['binary'].data[i,j] = 0.0

    def TS_initial(self):
        self.rand_array = torch.randn(10000000)
        self.ts_trick = self.params["ts_trick"]
        if self.type == 0 or self.type == 2:
            leng = self.dim
        elif self.type == 1 or self.type == 3:
            leng = self.dim*len(PRIMITIVES)
        if self.first_order == 1:
            fm_first_std_Linears = nn.ModuleList(
                    [nn.Linear(feature_size, 1, bias=False) for feature_size in self.feature_sizes[:self.dense[0]]])
            fm_first_std_embeddings = nn.ModuleList(
                    [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes[self.dense[0]:self.dense[0]+self.dense[1]]])
            fm_first_std_multi = nn.Embedding(self.dense[2], 1) 
            self.fm_first_std_models = fm_first_std_Linears.extend(fm_first_std_embeddings).append(fm_first_std_multi)

        fm_second_std_Linears = nn.ModuleList(
            [nn.Linear(feature_size, leng, bias=False) for feature_size in self.feature_sizes[:self.dense[0]]])
        fm_second_std_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, leng) for feature_size in self.feature_sizes[self.dense[0]:self.dense[0]+self.dense[1]]])
        fm_second_std_multi = nn.Embedding(self.dense[2], leng) 
        self.fm_second_std_models = fm_second_std_Linears.extend(fm_second_std_embeddings).append(fm_second_std_multi)

    def reparameterize(self, mu, std, alpha):
        std = torch.log(1 + torch.exp(std)).to(self.device)
        # v = torch.randn(batch, mu.shape[0], mu.shape[1]).to(self.device)
        v = self.rand_array[:std.numel()].reshape(std.shape).to(self.device)
        return (mu + alpha * std * v)

    def KL_distance(self, mean1, mean2, std1, std2):
        a = torch.log(std2/std1) + (std1*std1+(mean1-mean2)*(mean1-mean2))/2/std2/std2 - 1.0/2.0
        return torch.sum(a)

    def forward_ts(self, x, flag, weights=None, cal_kl=1):
        if weights == None:
            weights = self._arch_parameters['binary']
        X = x.reshape(x.shape[0], x.shape[1], 1)
        # if cal_kl==1:
        #     alpha = 1
        # else:
        #     alpha = self.alpha
        alpha = self.alpha
        out_first_order = 0
        if self.first_order == 1:
            for i, emb in enumerate(self.fm_first_order_models):
                if i < self.dense[0]:
                    Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.float)
                    X_mean = torch.sum(emb(Xi_tem).unsqueeze(1), 1)
                    if i < self.f_len and self.ts_trick==0:
                        out_first_order += X_mean
                    else:
                        X_std = torch.sum(self.fm_first_std_models[i](Xi_tem).unsqueeze(1), 1)
                        out_first_order += self.reparameterize(X_mean, X_std, alpha)
                elif i < self.dense[0]+self.dense[1]:
                    Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.long)
                    # print(Xi_tem.shape)
                    X_mean = torch.sum(emb(Xi_tem), 1)
                    if i < self.f_len and self.ts_trick==0:
                        out_first_order += X_mean
                    else:
                        X_std = torch.sum(self.fm_first_std_models[i](Xi_tem), 1)
                        out_first_order += self.reparameterize(X_mean, X_std, alpha)
                else:
                    for j in range(self.dense[2]):
                        Xi_tem = X[:, i+j, :].to(device=self.device, dtype=torch.long)
                        X_mean = Xi_tem*emb(torch.Tensor([j]).to(device=self.device, dtype=torch.long))
                        if i < self.f_len and self.ts_trick==0:
                            out_first_order += X_mean
                        else:
                            X_std = Xi_tem*self.fm_first_std_models[i](torch.Tensor([j]).to(device=self.device, dtype=torch.long))
                            out_first_order += self.reparameterize(X_mean, X_std, alpha)
        X_vector = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i < self.dense[0]:
                if self.params["calcu_dense"]==0:
                    continue
                else:
                    Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.float)
                    X_mean = emb(Xi_tem).unsqueeze(1)
                    if i < self.f_len and self.ts_trick==0:
                         X_vector.append(X_mean)
                    else:
                        X_std = self.fm_second_std_models[i](Xi_tem).unsqueeze(1)
                        X_vector.append(self.reparameterize(X_mean, X_std, alpha))
            elif i < self.dense[0]+self.dense[1]:
                Xi_tem = X[:, i, :].to(device=self.device, dtype=torch.long)
                X_mean = emb(Xi_tem)
                if i < self.f_len and self.ts_trick==0:
                    X_vector.append(X_mean)
                else:
                    X_std = self.fm_second_std_models[i](Xi_tem)
                    X_vector.append(self.reparameterize(X_mean, X_std, alpha))
            else:
                for j in range(self.dense[2]):
                        Xi_tem = X[:, i+j, :].to(device=self.device, dtype=torch.long)
                        X_mean = (Xi_tem*emb(torch.Tensor([j]).to(device=self.device, dtype=torch.long))).unsqueeze(1)
                        if i < self.f_len and self.ts_trick==0:
                            X_vector.append(X_mean)
                        else:
                            X_std = (Xi_tem*self.fm_second_std_models[i](torch.Tensor([j]).to(device=self.device, dtype=torch.long))).unsqueeze(1)
                            X_vector.append(self.reparameterize(X_mean, X_std, alpha))
        out_second_order = 0
        cnt = 0
        multi_hot_len = 0
        if self.dense[2] != 0:
            multi_hot_len = 1
        for i in range(len(X_vector)):
            for j in range(i):
                if i < len(self.feature_sizes)-multi_hot_len:
                    tmp = cnt
                elif j < len(self.feature_sizes)-multi_hot_len:
                    tmp = -len(self.feature_sizes)+j
                else:
                    tmp = -1
                if self.name != "SNAS":
                    out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp,:], flag, dim=self.dim, FC=self._FC, o_type=self.type)
                else:
                    out_second_order += MixedBinary(X_vector[i], X_vector[j], weights[tmp], flag, dim=self.dim, FC=self._FC, o_type=self.type)
                cnt += 1
                    
        out =  torch.sigmoid((out_second_order+out_first_order+self.bias).squeeze(1))
        if cal_kl == 0:
            return 0, out
        # print(out.shape,out_second_order.shape,out_first_order.shape)
        k = 0
        for i, emb in enumerate(self.fm_first_order_models):
            if i < self.f_len:
                continue
            k += self.KL_distance(emb.weight, 0*torch.ones_like(emb.weight), torch.log(1 + torch.exp(self.fm_first_std_models[i].weight)), 0.1*torch.ones_like(self.fm_first_std_models[i].weight))
        for i, emb in enumerate(self.fm_second_order_models):
            if i < self.f_len:
                continue
            k += self.KL_distance(emb.weight, 0*torch.ones_like(emb.weight), torch.log(1 + torch.exp(self.fm_second_std_models[i].weight)), 0.1*torch.ones_like(self.fm_second_std_models[i].weight))
        # print(k.shape)
        return k, out


class NAS(OFM):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__(feature_size, k, params, m_type)
        self.name = "NAS"

    def binarize(self):
        self._cache = self._arch_parameters['binary'].clone()
        max_index = [self._arch_parameters['binary'][i, :].argmax().item() 
                     for i in range(self.multi_len)]
        leng = len(self._arch_parameters['binary'][0,:])
        for i in range(self.multi_len):
            for j in range(leng):
                if j == max_index[i]:
                    self._arch_parameters['binary'].data[i,j] = 1.0
                else:
                    self._arch_parameters['binary'].data[i,j] = 0.0
        # print(self._arch_parameters['binary'])

    def recover(self):
        self._arch_parameters['binary'].data = self._cache
        del self._cache

    def step(self, x, labels_valid, criterion, arch_optimizer, other):
        # print(len(x),len(labels_valid))
        self.zero_grad()
        arch_optimizer.zero_grad()
        # binarize before forward propagation
        self.binarize()
        inferences = self(x, 1)
        loss = criterion(inferences, labels_valid)
        # loss = F.binary_cross_entropy(input=inferences,target=labels_valid,reduction='mean')
        loss.backward()
        # restore weight before updating
        self.recover()
        arch_optimizer.step()
        return loss

    def print(self):
        for name, i in self.named_parameters():
            print(name,i)
        return


class DSNAS(OFM):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0, args=None):
        super(DSNAS, self).__init__(feature_size, k, params, m_type)
        self.log_alpha = self._arch_parameters['binary']
        self.weights = Variable(torch.zeros_like(self.log_alpha))
        self.fix_arch_index = {}
        self.args = args
        self.name = "DSNAS"
    
    def binarize(self):
        pass

    def recover(self):
        return

    def fix_arch(self):
        if self.params["early_fix_arch"]:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.log_alpha.data[key, :] = value_lst[1]
            sort_log_alpha = torch.topk(F.softmax(self.log_alpha.data, dim=-1), 2)
            argmax_index = (sort_log_alpha[0][:, 0] - sort_log_alpha[0][:, 1] >= 0.3)
            for id in range(argmax_index.size(0)):
                if argmax_index[id] == 1 and id not in self.fix_arch_index.keys():
                    self.fix_arch_index[id] = [sort_log_alpha[1][id, 0].item(),
                                                self.log_alpha.detach().clone()[id, :]]


    def forward(self, x, flag, weights=None):
        if weights == None:
            weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim = -1).view(-1,1), 1)
        return super().forward(x, flag, weights)

    #only need training data for training architecture parameter and network parameters
    def step(self, x, labels, criterion, arch_optimizer, optimizer):
        error_loss = 0
        loss_alpha = 0
        self.fix_arch()
        arch_optimizer.zero_grad()
        optimizer.zero_grad()

        if self.params["early_fix_arch"]:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.weights[key, :].zero_()
                    self.weights[key, value_lst[0]] = 1

        self.weights = self._get_weights(self.log_alpha)
        cate_prob = F.softmax(self.log_alpha, dim=-1)
        self.cate_prob = cate_prob.clone().detach()
        loss_alpha = torch.log(
            (self.weights * F.softmax(self.log_alpha, dim=-1)).sum(-1)).sum()
        self.weights.requires_grad_()

        inference = self(x, 1, self.weights)
        
        error_loss = F.binary_cross_entropy(inference, labels.float())
        
        self.weights.grad = torch.zeros_like(self.weights)
        (error_loss + loss_alpha).backward()
        self.block_reward = self.weights.grad.data.sum(-1)
        self.log_alpha.grad.data.mul_(self.block_reward.view(-1, 1))
        arch_optimizer.step()
        optimizer.step()
        return error_loss

    def _get_weights(self, log_alpha):
        # if self.args.random_sample:
        #     uni = torch.ones_like(log_alpha)
        #     m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
        # else:
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()


class SNAS(OFM):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__(feature_size, k, params, m_type)
        self.arch_prob = [0 for _ in range(self.multi_len)]
        self.name = "SNAS"

    def binarize(self, temperature=0.00001):
        self.g_softmax(temperature)

    def recover(self):
        return 

    def forward(self, x, flag):
        return super().forward(x, flag, self.arch_prob)

    def g_softmax(self, temperature):
        self.temp = temperature
        for i in range(self.multi_len):
            # alpha = self._arch_parameters['binary'].data[i, :]
            # m = torch.nn.functional.gumbel_softmax(alpha, tau=temperature, hard=False, eps=1e-10, dim=-1)
            # m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            #     torch.tensor([temperature]).to(self.device) , alpha)
            # print("sam",m.sample(),"raw",self._arch_parameters['binary'].data[i, :])
            self.arch_prob[i] = torch.nn.functional.gumbel_softmax(self._arch_parameters['binary'][i, :],tau=temperature, hard=False, eps=1e-10, dim=-1)

    def step(self, x, labels_valid, criterion, arch_optimizer, temperature):
        self.zero_grad()
        arch_optimizer.zero_grad()
        self.g_softmax(temperature)
        inferences = self(x, 1)
        loss = criterion(inferences, labels_valid)
        # loss = F.binary_cross_entropy(input=inferences,target=labels_valid,reduction='mean')
        loss.backward()
        arch_optimizer.step()
        # for parms in self.arch_parameters():	
        #     print(parms,'-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
        return loss


class OFM_TS(OFM):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__(feature_size, k, params, m_type)
        self.TS_initial()


class NAS_TS(NAS):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__(feature_size, k, params, m_type)
        self.TS_initial()

    def step_ts(self, x, labels_valid, criterion, arch_optimizer, other):
        # print(len(x),len(labels_valid))
        self.zero_grad()
        arch_optimizer.zero_grad()
        # binarize before forward propagation
        self.binarize()
        _, inferences = self.forward_ts(x, 1, cal_kl=0)
        loss = criterion(inferences, labels_valid)
        # loss = F.binary_cross_entropy(input=inferences,target=labels_valid,reduction='mean')
        loss.backward()
        # restore weight before updating
        self.recover()
        arch_optimizer.step()
        return loss

class SNAS_TS(SNAS):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__(feature_size, k, params, m_type)
        self.TS_initial()


class DSNAS_TS(DSNAS):
    def __init__(self, feature_size=None, k=None, params=None, m_type=0):
        super().__init__(feature_size, k, params, m_type)
        self.TS_initial()


class Linucb(nn.Module):
    def __init__(self, n=1, cnum=1, device='cuda'):
        super().__init__()
        self.alpha = 0.1
        self.n =n
        self.theta = nn.Parameter(torch.randn(cnum, n).to(device))
        self.A_inv = nn.Parameter(torch.randn(cnum, n, n).to(device))

    def forward(self, x):
        ind = x[:,0].cpu().numpy().astype(int)
        feature = x[:,1:].reshape(len(x),self.n)
        mean = torch.mul(feature, self.theta[ind]).sum(1, keepdim=True)
        fe1 = feature.reshape(len(x),1,self.n)
        fe2 = feature.reshape(len(x),self.n,1)
        std = self.alpha*torch.sqrt(torch.bmm(torch.bmm(fe1,self.A_inv[ind]),fe2).reshape(len(x),1))
        return mean + std

    def print():
        return
