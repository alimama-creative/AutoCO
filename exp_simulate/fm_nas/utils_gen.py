from train import Core
from train_arch import Train_Arch
import argparse
import pickle
import numpy as np
import torch


dataset_name = "data_nas"
def gen_simulation_ctr():
    c_list = None
    with open("raw_data/"+dataset_name+"/creative_list.pkl", 'rb') as f:
        c_list = pickle.load(f)
    c_list = np.array(c_list, dtype=np.float32)
    item_candi = None
    with open("/raw_data/"+dataset_name+"/item_candidates.pkl", 'rb') as f:
        item_candi = pickle.load(f)
    f_list = None
    with open("raw_data/"+dataset_name+"/feature_list.pkl", 'rb') as f:
        f_list = pickle.load(f)

    params = {"device": "cuda",
            "alpha": 0.0,
            "feature_size": "raw_data/"+dataset_name+"/feature_size.pkl",
            "model_struct": 3,
            "model_nas": "nas",
            }

    framework = Train_Arch(dim=8,
                           weight_decay=0.001,
                           data_size= -1, 
                           device="cuda",
                           epoch=50,
                           params=params)
    model_path = "raw_data/data_nas/rand_0.15_3/model/fm_nas.pt"
    framework.load_model(model_path)
    framework.model.genotype()
    # print(framework.model.named_parameters)
    # for i in range(len(framework.model._arch_parameters["binary"])):
    #     framework.model._arch_parameters["binary"][i, np.random.randint(0,5)] = 1
    arch = [framework.model._arch_parameters['binary'][i, :].argmax().item() 
                        for i in range(len(framework.model._arch_parameters['binary']))]
    print(arch)
    framework.model.genotype()
    f_len = len(f_list[0]) 
    leng = len(f_list[0])+len(c_list[0])
    num = 0
    for ind in range(len(f_list)):
        num += len(item_candi[ind])        
    candidates = np.zeros((num, leng),dtype=np.float32)
    cnt = 0
    for ind in range(len(f_list)):
        t = np.zeros(leng)
        t[0:f_len] = np.array(f_list[ind], dtype=np.float32)
        # print(item_candi[ind])
        for c_feature in item_candi[ind]:
            t[f_len:] = c_list[c_feature]
            candidates[cnt] = t
            cnt += 1
    # print(candidates.shape)
    rank = framework.result(candidates)
    # print(rank)
    # for name,par in framework.model.named_parameters():
    #     print(name,par)
    ctr_list = []
    item_ctr = {}
    cnt = 0
    for ind in range(len(f_list)):
        item_ctr[ind] = []
        for c_feature in item_candi[ind]:
            item_ctr[ind].append(rank[cnt])
            ctr_list.append(rank[cnt])
            cnt += 1 
    ctr_list = sorted(ctr_list)
    print('low: ',ctr_list[:10])
    print('high: ',ctr_list[-10:])
    radio = 0
    for ind in range(len(f_list)):
        # print(item_ctr[ind])
        c = sorted(item_ctr[ind])
        a = c[-1]/c[int(len(c)/2)] - 1
        radio += a
    radio /= len(f_list)
    print(radio)

    k = 2
    cnt = 0
    res = []
    for name, parameter in framework.model.named_parameters():
        parameter[:] = parameter * 1.3
        cnt += 1
        res.append(parameter)
    rank = framework.result(candidates)
    ctr_list = []
    item_ctr = {}
    cnt = 0
    for ind in range(len(f_list)):
        item_ctr[ind] = []
        for c_feature in item_candi[ind]:
            item_ctr[ind].append(rank[cnt])
            ctr_list.append(rank[cnt])
            cnt += 1
    ctr_list = sorted(ctr_list)
    print('low: ',ctr_list[0:10])
    print('high: ',ctr_list[-10:])
    radio = 0
    for ind in range(len(f_list)):
        c = sorted(item_ctr[ind])
        a = c[-1]/c[int(len(c)/2)] - 1
        radio += a
    radio /= len(f_list)
    print(radio)
    



if __name__ == "__main__":
    gen_simulation_ctr()