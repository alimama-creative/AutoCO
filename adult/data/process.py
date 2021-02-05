import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data1 = pd.read_csv("./adult.data",header=None)
data2 = pd.read_csv("./adult.test",header=None)
data = pd.concat([data1, data2], axis=0)
data.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain",
                "capital-loss", "hours-per-week", "native-country", "income"]

data_category = data[["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]]

data_category["income"] = data_category["income"].map({' <=50K': '<=50k', ' >50K': '>50k', ' <=50K.': '<=50k', ' >50K.': '>50k'})
def encode(data, column):
    label = sorted(data[column].unique(), key=lambda x:x)
    label_indices = list(range(len(label)))
    label_dict = dict(zip(label, label_indices))
    data[column] = data[column].map(label_dict)

for name in data_category.columns:
    encode(data_category, name)

train_X = data_category.iloc[:,:-1].values
train_y = data_category.iloc[:,-1].values
clf = LogisticRegression(random_state=0).fit(train_X, train_y)
predicted = clf.predict_proba(train_X)
# reward: 36925
print("AUC score: ", roc_auc_score(train_y,predicted[:,1]))

data_category["eat_reward"] = data_category["income"].map(lambda x: 1 if x==0 else 0)
data_category["noteat_reward"] = data_category["income"].map(lambda x: 1 if x==1 else 0)
data_category.to_csv("./data.csv", index=False)

N = 100000
sample_indices = np.random.randint(0, data_category.shape[0], size=N)
data_sample = data_category.loc[sample_indices]
data_sample.to_csv("./data_sample.csv", index=False)
#AUC: 0.7279