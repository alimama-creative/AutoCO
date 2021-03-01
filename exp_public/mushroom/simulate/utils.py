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

class Mushroom(Dataset):
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
            sample["cap-shape"] = self.data.iloc[index]["cap-shape"]
            sample["cap-surface"] = self.data.iloc[index]["cap-surface"]
            sample["cap-color"] = self.data.iloc[index]["cap-color"]
            sample["bruises"] = self.data.iloc[index]["bruises"]
            sample["odor"] = self.data.iloc[index]["odor"]
            sample["gill-attachment"] = self.data.iloc[index]["gill-attachment"]
            sample["gill-spacing"] = self.data.iloc[index]["gill-spacing"]
            sample["gill-size"] = self.data.iloc[index]["gill-size"]
            sample["gill-color"] = self.data.iloc[index]["gill-color"]
            sample["stalk-shape"] = self.data.iloc[index]["stalk-shape"]
            sample["stalk-root"] = self.data.iloc[index]["stalk-root"]
            sample["stalk-surface-above-ring"] = self.data.iloc[index]["stalk-surface-above-ring"]
            sample["stalk-surface-below-ring"] = self.data.iloc[index]["stalk-surface-below-ring"]
            sample["stalk-color-above-ring"] = self.data.iloc[index]["stalk-color-above-ring"]
            sample["stalk-color-below-ring"] = self.data.iloc[index]["stalk-color-below-ring"]
            sample["veil-type"] = self.data.iloc[index]["veil-type"]
            sample["veil-color"] = self.data.iloc[index]["veil-color"]
            sample["ring-number"] = self.data.iloc[index]["ring-number"]
            sample["ring-type"] = self.data.iloc[index]["ring-type"]
            sample["spore-print-color"] = self.data.iloc[index]["spore-print-color"]
            sample["population"] = self.data.iloc[index]["population"]
            sample["habitat"] = self.data.iloc[index]["habitat"]
            eat_reward = self.data.iloc[index]["eat_reward"]
            noteat_reward = self.data.iloc[index]["noteat_reward"]
            sample["label"] = torch.Tensor([noteat_reward, eat_reward])
            #sample["label"] = torch.Tensor([eat_reward, noteat_reward])
        else:
            cols = list(self.data.columns)
            _ = cols.remove("eat_reward")
            _ = cols.remove("noteat_reward")
            data2 = self.one_hot(self.data, cols)
            sample["feature"] = torch.Tensor(data2.iloc[index][:])
            eat_reward = self.data.iloc[index]["eat_reward"]
            noteat_reward = self.data.iloc[index]["noteat_reward"]
            sample["label"] = torch.Tensor([eat_reward, noteat_reward])
        return sample

def get_data_queue(args, dummpy):
    print(args.dataset)
    if args.dataset == 'mushroom':
        train_data = "../data/data_sample.csv"
        train_dataset = Mushroom(train_data, dummpy)
        train_queue = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True)
        return train_queue
    else:
        return None

class Mushroom2(Dataset):
    def __init__(self, contexts, pos_weights):
        self.data = contexts
        self.pos_weights = pos_weights

    def __len__(self):
        return self.data["label"].shape[0]

    def __getitem__(self, index):
        sample = {}
        sample["cap-shape"] = self.data["cap-shape"][index]
        sample["cap-surface"] = self.data["cap-surface"][index]
        sample["cap-color"] = self.data["cap-color"][index]
        sample["bruises"] = self.data["bruises"][index]
        sample["odor"] = self.data["odor"][index]
        sample["gill-attachment"] = self.data["gill-attachment"][index]
        sample["gill-spacing"] = self.data["gill-spacing"][index]
        sample["gill-size"] = self.data["gill-size"][index]
        sample["gill-color"] = self.data["gill-color"][index]
        sample["stalk-shape"] = self.data["stalk-shape"][index]
        sample["stalk-root"] = self.data["stalk-root"][index]
        sample["stalk-surface-above-ring"] = self.data["stalk-surface-above-ring"][index]
        sample["stalk-surface-below-ring"] = self.data["stalk-surface-below-ring"][index]
        sample["stalk-color-above-ring"] = self.data["stalk-color-above-ring"][index]
        sample["stalk-color-below-ring"] = self.data["stalk-color-below-ring"][index]
        sample["veil-type"] = self.data["veil-type"][index]
        sample["veil-color"] = self.data["veil-color"][index]
        sample["ring-number"] = self.data["ring-number"][index]
        sample["ring-type"] = self.data["ring-type"][index]
        sample["spore-print-color"] = self.data["spore-print-color"][index]
        sample["population"] = self.data["population"][index]
        sample["habitat"] = self.data["habitat"][index]
        sample["label"] = self.data["label"][index]
        sample["pos_weights"] = self.pos_weights[index]
        return sample

def get_data_queue_bandit(args, contexts, pos_weights):
    train_dataset = Mushroom2(contexts, pos_weights)
    train_queue = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True)
    return train_queue
