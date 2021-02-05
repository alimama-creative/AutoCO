import numpy as np
import pandas as pd

data = pd.read_csv("data.txt", header=None)
data.columns = ["label", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
N = 50000
sample_indices = np.random.randint(0, data.shape[0], size=N)

data_sample = data.loc[sample_indices]

def encode(data, column):
    label = sorted(data[column].unique(), key=lambda x:x)
    label_indices = list(range(len(label)))
    label_dict = dict(zip(label, label_indices))
    data.loc[:, column] = data[column].map(label_dict)

for name in data_sample.columns:
    encode(data_sample, name)

def reward(x):
    if x == 0:
        return 5
    elif x == 1:
        if np.random.random() >= 0.5:
            return 5
        else:
            return -35
    else:
        return -1

#revised the label, 0 is good,1 is poison
data_sample["eat_reward"] = data_sample["label"].map(lambda x: reward(x))
data_sample["noteat_reward"] = 0
data_sample = data_sample[["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat", "eat_reward", "noteat_reward"]]
data_sample.to_csv("./data_sample.csv", index=False)
