from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import random
import re
import json
import pandas as pd

tfs_feature_file = 'raw_data/tfs_features2.txt'
log_file = "raw_data/log_online.txt"
cnt = 0

X_template = []
X_cate1 = []
X_cate2 = []
X_encode = []
X_encode_0 = []
X_encode_1 = []
X_encode_2 = []
X_encode_3 = []
X_pic = []
Y = []


tfs_feat = {}
with open(tfs_feature_file, 'r') as f:
    for line in f:
        seq = line.split()
        tfs = seq[0]
        if tfs not in tfs_feat.keys():
            # tfs_feat[tfs] = seq[1]
            tt = [float(s) for s in seq[1].split(',')]
            mu = np.mean(tt)
            std = np.std(tt, ddof=1)
            tfs_feat[tfs] = [float(s-mu)/std for s in tt]


with open(log_file, 'r') as ff:
    for line in ff:
        line = line.split('\n')[0]
        line = line.split('//')

        tfs = line[3]
        tid = line[5]
        clk = 1 if line[1][0] is not '(' else 0
        # if clk==0 and np.random.rand>0.8:
        #     continue
        encode = [int(e) for e in line[2].split()]
        encode_len = len(encode)
        # print(encode_len)
        # print(encode)
        cate1 = line[4]

        if tfs in tfs_feat.keys():
            tfs_v = tfs_feat[tfs]
            X_template.append(int(tid))
            X_cate1.append(int(cate1))
            X_encode.append(encode)
            X_encode_0.append(encode[0])
            X_encode_1.append(encode[1])
            X_encode_2.append(encode[2])
            X_encode_3.append(encode[3])
            X_pic.append(tfs_v)
            Y.append(clk)
            cnt += 1
        # if cnt >1010000:
        #     break
        elif tfs not in tfs_feat.keys():
            print("Error ! No tfs!!!", tfs)

print("finish loading")
le = LabelEncoder()
X_template = le.fit_transform(X_template)
X_cate1 = le.fit_transform(X_cate1)
X_encode_0 = le.fit_transform(X_encode_0)
X_encode_1 = le.fit_transform(X_encode_1)
X_encode_2 = le.fit_transform(X_encode_2)
X_encode_3 = le.fit_transform(X_encode_3)
# scale = StandardScaler()
# X_pic = scale.fit_transform(X_pic)
# length = len(Y)
length = len(Y)
data = []
ind  = 0
item_dict = {}
distinct_encode = []
dict_encode = {}
item_encode = {}
feature_list = []
pro_list = []
for x in range(length):
    if x%20000==0:
        print(x,' has generated')
    data.append([])
    data[x].extend(X_pic[x])
    data[x].append(X_cate1[x])
    data[x].append(X_template[x])

    if X_encode[x] not in distinct_encode:
        distinct_encode.append(X_encode[x])
        dict_encode[','.join([str(e) for e in X_encode[x]])] = len(distinct_encode)-1
    key = ','.join([str(e) for e in data[x]])
    if key not in item_dict.keys():
        item_dict[key] = ind
        feature_list.append(data[x].copy())
        pro_list.append(0)
        item_encode[ind] = []
        ind += 1
    pro_list[item_dict[key]]+=1
    t = dict_encode[','.join([str(e) for e in X_encode[x]])]
    if t not in item_encode[item_dict[key]]:
        item_encode[item_dict[key]].append(t)

    data[x].append(X_encode_0[x])
    data[x].append(X_encode_1[x])
    data[x].append(X_encode_2[x])
    data[x].append(X_encode_3[x])
    # print(data[x])
    # data[x].insert(0, item_dict[key])
    data[x].append(Y[x])

for i in range(len(pro_list)):
    pro_list[i]/=float(length)


out_file = 'dataset/dataset_1400.pkl'
data_p = pd.DataFrame(data)
# # print(data_p.shape)
data_p.to_pickle(out_file)


# out_file = "data_nas/log.pkl"
out_file_1 = "./creative_list.pkl"
out_file_2 = './item_candidates.pkl'
out_file_3 = './feature_list.pkl'
out_file_4 = './pro_list.pkl'
import pickle
# with open(out_file, 'wb') as d:
#     pickle.dump(data, d)
with open(out_file_1, 'wb') as d:
    pickle.dump(distinct_encode, d)
with open(out_file_2, 'wb') as d:
    pickle.dump(item_encode, d)
with open(out_file_3, 'wb') as d:
    pickle.dump(feature_list, d)
with open(out_file_4, 'wb') as d:
    pickle.dump(pro_list, d)