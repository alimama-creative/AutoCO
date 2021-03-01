import numpy as np
import pandas as pd
import os
import os.path
import sys
import shutil
import torch
import torch.nn as nn
import torch.utils
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from models import PRIMITIVES_BINARY

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def sample_arch():
    arch = {}
    arch['mlp'] = {}
    arch['mlp']['p'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1))
    arch['mlp']['q'] = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1))
    arch['binary'] = PRIMITIVES_BINARY[np.random.randint(len(PRIMITIVES_BINARY))]
    return arch

class Adult(Dataset):
    def __init__(self, root_dir, dummpy):
        self.data = pd.read_csv(root_dir)
        self.dummpy = dummpy

    def __len__(self):
        return self.data.shape[0]

    def one_hot(self, df, cols):
        res = []
        for col in cols:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            res.append(dummies)
        df = pd.concat(res, axis=1)
        return df

    def __getitem__(self, index):
        sample = {}
        if not self.dummpy:
            sample["workclass"] = self.data.iloc[index]["workclass"]
            sample["education"] = self.data.iloc[index]["education"]
            sample["marital-status"] = self.data.iloc[index]["marital-status"]
            sample["occupation"] = self.data.iloc[index]["occupation"]
            sample["relationship"] = self.data.iloc[index]["relationship"]
            sample["race"] = self.data.iloc[index]["race"]
            sample["sex"] = self.data.iloc[index]["sex"]
            sample["native-country"] = self.data.iloc[index]["native-country"]

            eat_reward = self.data.iloc[index]["eat_reward"]
            noteat_reward = self.data.iloc[index]["noteat_reward"]
            sample["label"] = torch.Tensor([eat_reward, noteat_reward])
        else:
            cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
            data2 = self.one_hot(self.data, cols)
            sample["feature"] = torch.Tensor(data2.iloc[index][:])
            eat_reward = self.data.iloc[index]["eat_reward"]
            noteat_reward = self.data.iloc[index]["noteat_reward"]
            sample["label"] = torch.Tensor([eat_reward, noteat_reward])
        return sample

def get_data_queue(args, dummpy):
    print(args.dataset)
    if args.dataset == 'Adult':
        train_data = "../data/data.csv"
        train_dataset = Adult(train_data, dummpy)
        train_queue = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
        return train_queue
    else:
        return None

class Adult2(Dataset):
    def __init__(self, contexts, pos_weights):
        self.data = contexts
        self.pos_weights = pos_weights

    def __len__(self):
        return self.data["label"].shape[0]

    def __getitem__(self, index):
        sample = {}
        sample["workclass"] = self.data["workclass"][index]
        sample["education"] = self.data["education"][index]
        sample["marital-status"] = self.data["marital-status"][index]
        sample["occupation"] = self.data["occupation"][index]
        sample["relationship"] = self.data["relationship"][index]
        sample["race"] = self.data["race"][index]
        sample["sex"] = self.data["sex"][index]
        sample["native-country"] = self.data["native-country"][index]
        sample["label"] = self.data["label"][index]
        sample["pos_weights"] = self.pos_weights[index]
        return sample

def get_data_queue_bandit(args, contexts, pos_weights):
    train_dataset = Adult2(contexts, pos_weights)
    train_queue = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True)
    return train_queue