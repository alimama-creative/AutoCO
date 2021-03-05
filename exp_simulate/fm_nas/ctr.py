import numpy as np
import pandas as pd
import torch.utils.data
import itertools
import tqdm
import time


def get_index(dataset, dense_list=None):
    emb_index = []
    cnt = 0
    dim = 0
    for i in range(len(dataset)):
        emb_index.append(dict())
        if i in dense_list:
            dim += 1
            continue
        ind = dataset[i].value_counts().index
        size = len(ind)
        for key in ind:
            emb_index[i][key] = cnt
            cnt += 1
            dim += 1
    return emb_index, dim

def get_field_info(dataset, dense=[0,0]):
    feature_size = []
    for i in range(dense[1]+dense[0]):
        if i < dense[0]:
            feature_size.append(1)
            continue
        ind = dataset[i].value_counts().index
        print(ind)
        size = len(ind)
        feature_size.append(size)
    return feature_size

class DirectDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_r):
        dataset = np.array(dataset_r)
        print(len(dataset))
        self.items = dataset[:, 0:-1].astype(np.float32)
        self.n = len(self.items[0])
        self.targets = dataset[:, -1].astype(np.float32)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


class TfsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num=-1):
        df = pd.read_pickle(dataset_path)
        data = df.to_numpy()
        self.items = data[:num, 0:-1].astype(np.float32)
        (a, b) = np.shape(self.items)
        print(a, b)
        self.n = b
        print(data[:,-1])
        self.targets = self.__preprocess_target(data[:num, -1].astype(np.float32)).astype(np.float32)
        # self.field_dims = np.max(self.items, axis50=0) + 1
        # self.field_dims = np.array([2]*b)
        self.user_field_idx = np.array((0,), dtype=np.float32)
        self.item_field_idx = np.array((1,), dtype=np.float32)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 0] = 0
        target[target > 0] = 1
        return target


class DenseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num=-1, dense=[10,6,0], flag=0):
        print(dataset_path)
        df = pd.read_pickle(dataset_path)
        data = df.to_numpy()
        # np.random.shuffle(data)
        if flag == 0:
            self.items = data[:num, 0:-1].astype(np.float32)
            self.targets = data[:num, -1].astype(np.float32)
        else:
            self.items = data[-1-num:-1, 0:-1].astype(np.float32)
            self.targets = data[-1-num:-1, -1].astype(np.float32)
        (a, b) = np.shape(self.items)
        print(a, b)
        print(data[:,-1])
        
        # self.embedding_ind, dim = get_index(self.items, [0])
        # self.feature_size = get_field_info(df, dense)
        # get_field_info(df, dense)
        # self.feature_size.append(dense[2])
        # self.feature_size[4] = 3952
        # import pickle
        # with open(path, 'wb') as d:
        #     pickle.dump(self.feature_size, d)
        # print(self.feature_size)
        # self.n = dim

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


if __name__=="__main__":
    ds = DenseDataset('raw_data/data_movie_log/log.pkl')
