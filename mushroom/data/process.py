import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.txt", header=None)
data.columns = ["label", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
def encode(data, column):
    label = sorted(data[column].unique(), key=lambda x:x)
    label_indices = list(range(len(label)))
    label_dict = dict(zip(label, label_indices))
    data[column] = data[column].map(label_dict)

for name in data.columns:
    encode(data, name)

X, test_X, y, test_y = train_test_split(data.iloc[:,1:].values, data.iloc[:,0].values, test_size=0.20, random_state=42)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.10, random_state=42)

train = pd.DataFrame(data=np.hstack([train_y.reshape(-1,1), train_X]), columns=data.columns)
train.to_csv("./train.csv", index=False)

val = pd.DataFrame(data=np.hstack([val_y.reshape(-1,1), val_X]), columns=data.columns)
val.to_csv("./val.csv", index=False)

test = pd.DataFrame(data=np.hstack([test_y.reshape(-1,1), test_X]), columns=data.columns)
test.to_csv("./test.csv", index=False)
# # train = data.iloc[:6513] #6513
# # val = data.iloc[6000:6513]
# # test = data.iloc[6513:] #1611
#
# # X = train.iloc[:,1:].values
# # y = train.iloc[:,0].values
# clf = LogisticRegression(random_state=0).fit(train_X, train_y)
#
# # test_X = test.iloc[:,1:].values
# # test_y = test.iloc[:,0].values
# predicted = clf.predict_proba(test_X)
#
# print("AUC score: ", roc_auc_score(test_y,predicted[:,1]))
# #0.980
