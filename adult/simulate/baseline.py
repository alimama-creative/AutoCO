import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import utils
from collections import Counter
from torch.distributions.multivariate_normal import MultivariateNormal

PRIMITIVES_BINARY = ['plus', 'multiply', 'max', 'min', 'concat']
PRIMITIVES_NAS = [0, 2, 4, 8, 16]
SPACE_NAS = pow(len(PRIMITIVES_NAS), 5)
OPS = {
    'plus': lambda p, q: p + q,
    'multiply': lambda p, q: p * q,
    'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0],
    'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0],
    'concat': lambda p, q: torch.cat([p, q], dim=-1),
    'norm_0': lambda p: torch.ones_like(p),
    'norm_0.5': lambda p: torch.sqrt(torch.abs(p) + 1e-7),
    'norm_1': lambda p: torch.abs(p),
    'norm_2': lambda p: p ** 2,
    'I': lambda p: torch.ones_like(p),
    '-I': lambda p: -torch.ones_like(p),
    'sign': lambda p: torch.sign(p),
}

class Virtue_v(nn.Module):

    def __init__(self, embedding_dim, reg, embedding_num=12, ofm=False, first_order=False):
        super(Virtue_v, self).__init__()
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.embedding_mean = nn.ModuleDict({})
        self.embedding_std = nn.ModuleDict({})
        if first_order:
            self.embedding_first_order = nn.ModuleDict({})
        self.columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

        for name in self.columns:
            self.embedding_mean[name] = nn.Embedding(embedding_num, embedding_dim)
            self.embedding_std[name] = nn.Embedding(embedding_num, embedding_dim)
            if first_order:
                self.embedding_first_order[name] = nn.Embedding(embedding_num, 1)

        self.embedding_action = nn.Embedding(2, embedding_dim)
        if first_order:
            self.embedding_action_first_order = nn.Embedding(2, 1)

class Virtue(nn.Module):

    def __init__(self, embedding_dim, reg, embedding_num=12, ofm=False, first_order=False):
        super(Virtue, self).__init__()
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.embedding_all = nn.ModuleDict({})
        if first_order:
            self.embedding_first_order = nn.ModuleDict({})
        self.columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

        for name in self.columns:
            self.embedding_all[name] = nn.Embedding(embedding_num, embedding_dim)
            if first_order:
                self.embedding_first_order[name] = nn.Embedding(embedding_num, 1)

        self.embedding_action = nn.Embedding(2, embedding_dim)
        if first_order:
            self.embedding_action_first_order = nn.Embedding(2, 1)

class FM_v(Virtue_v):
    """
    FM with EE
    """

    def __init__(self, embedding_dim, reg, args):
        super(FM_v, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num, first_order=args.first_order)
        self.args = args
        self._initialize_alphas()
        #initialize contextual infos
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.log_alpha = torch.Tensor([1, 1, 1, 1, 1.0])
        self.weights = Variable(torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
        self.rand_array = torch.randn(3000000)

    def reparameterize(self, mu, std):
        std = torch.log(1 + torch.exp(std))
        v = self.rand_array[:std.numel()].reshape(std.shape)
        return (mu + std * v * 0.01)

    def KL_distance(self, mean1, mean2, std1, std2):
        a = torch.log(std2 / std1) + (std1 * std1 + (mean1 - mean2) * (mean1 - mean2)) / 2 / std2 / std2 - 1.0 / 2.0
        return torch.sum(a)

    def recommend(self, features):
        self.eval()
        inferences = 0
        inferences_0 = 0
        inferences_1 = 0
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    name1_embedding_mean = self.embedding_mean[name1](features[name1])
                    name1_embedding_std = self.embedding_std[name1](features[name1])
                    name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                    name2_embedding_mean = self.embedding_mean[name2](features[name2])
                    name2_embedding_std = self.embedding_std[name2](features[name2])
                    name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    #inferences_0 += torch.sum(name1_embedding*name2_embedding, dim=1, keepdim=True)
                    #inferences_1 += torch.sum(name1_embedding*name2_embedding, dim=1, keepdim=True)
                    inferences += torch.sum(name1_embedding*name2_embedding, dim=1, keepdim=True)
        inferences_0 = 0  # inferences.clone()  # action 0
        inferences_1 = 0  # inferences.clone()  # action_1
        #features with action
        for name1 in self.columns:
            name1_embedding_mean = self.embedding_mean[name1](features[name1])
            name1_embedding_std = self.embedding_std[name1](features[name1])
            name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)

            inferences_0 += torch.sum(name1_embedding * self.embedding_action(torch.zeros_like(features[name1]).long()), dim=1, keepdim=True)
            inferences_1 += torch.sum(name1_embedding * self.embedding_action(torch.ones_like(features[name1]).long()), dim=1, keepdim=True)
            if self.args.first_order:
                name1_embedding_first_order = self.embedding_first_order[name1](features[name1])
                inferences_0 += name1_embedding_first_order
                inferences_1 += name1_embedding_first_order
        if self.args.first_order:
            inferences_0 += self.embedding_action_first_order(torch.zeros_like(features[name1]).long())
            inferences_1 += self.embedding_action_first_order(torch.ones_like(features[name1]).long())

        inferences = torch.cat([inferences_0, inferences_1], dim=1)
        pos_weights = torch.zeros_like(features["label"])
        max_index = torch.argmax(inferences, dim=1)
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        self.add_batch(features, pos_weights)
        return reward

    def add_batch(self, features, pos_weights):
        for index in features:
            if index in self.contexts:
                temp = self.contexts[index]
                self.contexts[index] = torch.cat([temp, features[index]], dim=0)
            else:
                self.contexts[index] = features[index]
        self.pos_weights = torch.cat([self.pos_weights, pos_weights], dim=0)

    def step(self, optimizer, arch_optimizer, epoch):
        self.train()
        losses = []
        train_bandit = utils.get_data_queue_bandit(self.args, self.contexts, self.pos_weights)
        if epoch < self.args.search_epoch:
            train_epoch = (epoch+1)*5
        else:
            train_epoch = 1
        for k in range(train_epoch):
            for step, features in enumerate(train_bandit):
                optimizer.zero_grad()
                output, error_loss, loss_alpha = self.forward(features)
                losses.append(error_loss.cpu().detach().item())
                optimizer.step()
        print(self.embedding_mean["workclass"](torch.LongTensor([[1]])))
        return np.mean(losses)

    def forward(self, features):
        regs = 0
        inferences = 0
        inferences_0 = 0
        inferences_1 = 0
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    name1_embedding_mean = self.embedding_mean[name1](features[name1])
                    name1_embedding_std = self.embedding_std[name1](features[name1])
                    name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                    name2_embedding_mean = self.embedding_mean[name2](features[name2])
                    name2_embedding_std = self.embedding_std[name2](features[name2])
                    name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    #inferences_0 += torch.sum(name1_embedding * name2_embedding, dim=1, keepdim=True)
                    #inferences_1 += torch.sum(name1_embedding * name2_embedding, dim=1, keepdim=True)
                    inferences += torch.sum(name1_embedding * name2_embedding, dim=1, keepdim=True)
        # features with action
        inferences_0 = 0  # inferences.clone()  # action 0
        inferences_1 = 0  # inferences.clone()  # action_1
        for name1 in self.columns:
            name1_embedding_mean = self.embedding_mean[name1](features[name1])
            name1_embedding_std = self.embedding_std[name1](features[name1])
            name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
            inferences_0 += torch.sum(name1_embedding * self.embedding_action(torch.zeros_like(features[name1]).long()),
                                      dim=1, keepdim=True)
            inferences_1 += torch.sum(name1_embedding * self.embedding_action(torch.ones_like(features[name1]).long()),
                                      dim=1, keepdim=True)

            if self.args.first_order:
                name1_embedding_first_order = self.embedding_first_order[name1](features[name1])
                inferences_0 += name1_embedding_first_order
                inferences_1 += name1_embedding_first_order
        if self.args.first_order:
            inferences_0 += self.embedding_action_first_order(torch.zeros_like(features[name1]).long())
            inferences_1 += self.embedding_action_first_order(torch.ones_like(features[name1]).long())
        inferences = torch.cat([inferences_0, inferences_1], dim=1)
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        kl = 0
        for name in self.columns:
            kl += self.KL_distance(self.embedding_mean[name].weight,
                                   0 * torch.ones_like(self.embedding_mean[name].weight),
                                   torch.log(1 + torch.exp(self.embedding_std[name].weight)),
                                   0.1 * torch.ones_like(self.embedding_std[name].weight))

        (weighted_loss + kl/features["label"].shape[0]).backward()
        return inferences, weighted_loss, 0

    def genotype(self):
        return "FM_v", 0


class FM(Virtue):
    """
    FM without EE
    """

    def __init__(self, embedding_dim, reg, args):
        super(FM, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num, first_order=args.first_order)
        self.args = args
        self._initialize_alphas()
        #initialize contextual infos
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.log_alpha = torch.Tensor([1, 1, 1, 1, 1.0])
        self.weights = Variable(torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
        self.rand_array = torch.randn(3000000)

    def recommend(self, features):
        self.eval()
        inferences = 0
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    name1_embedding = self.embedding_all[name1](features[name1])
                    name2_embedding = self.embedding_all[name2](features[name2])
                    inferences += torch.sum(name1_embedding*name2_embedding, dim=1, keepdim=True)
        inferences_0 = 0 #inferences.clone()  # action 0
        inferences_1 = 0 #inferences.clone()  # action_1

        #features with action
        for name1 in self.columns:
            name1_embedding = self.embedding_all[name1](features[name1])
            inferences_0 += torch.sum(name1_embedding * self.embedding_action(torch.zeros_like(features[name1]).long()), dim=1, keepdim=True)
            inferences_1 += torch.sum(name1_embedding * self.embedding_action(torch.ones_like(features[name1]).long()), dim=1, keepdim=True)
            if self.args.first_order:
                name1_embedding_first_order = self.embedding_first_order[name1](features[name1])
                inferences_0 += name1_embedding_first_order
                inferences_1 += name1_embedding_first_order
        if self.args.first_order:
            inferences_0 += self.embedding_action_first_order(torch.zeros_like(features[name1]).long())
            inferences_1 += self.embedding_action_first_order(torch.ones_like(features[name1]).long())

        inferences = torch.cat([inferences_0, inferences_1], dim=1)
        pos_weights = torch.zeros_like(features["label"])
        max_index = torch.argmax(inferences, dim=1)
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        self.add_batch(features, pos_weights)
        return reward

    def add_batch(self, features, pos_weights):
        for index in features:
            if index in self.contexts:
                temp = self.contexts[index]
                self.contexts[index] = torch.cat([temp, features[index]], dim=0)
            else:
                self.contexts[index] = features[index]
        self.pos_weights = torch.cat([self.pos_weights, pos_weights], dim=0)

    def step(self, optimizer, arch_optimizer, epoch):
        self.train()
        losses = []
        train_bandit = utils.get_data_queue_bandit(self.args, self.contexts, self.pos_weights)
        if epoch < self.args.search_epoch:
            train_epoch = (epoch+1)*5
        else:
            train_epoch = 1
        for k in range(train_epoch):
            for step, features in enumerate(train_bandit):
                optimizer.zero_grad()
                output, error_loss, loss_alpha = self.forward(features)
                losses.append(error_loss.cpu().detach().item())
                optimizer.step()
        print(self.embedding_all["workclass"](torch.LongTensor([[1]])))
        return np.mean(losses)

    def forward(self, features):
        regs = 0
        inferences = 0
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    name1_embedding = self.embedding_all[name1](features[name1])
                    name2_embedding = self.embedding_all[name2](features[name2])
                    inferences += torch.sum(name1_embedding * name2_embedding, dim=1, keepdim=True)
        # features with action
        inferences_0 = 0 #inferences.clone()  # action 0
        inferences_1 = 0 #inferences.clone()  # action_1
        for name1 in self.columns:
            name1_embedding = self.embedding_all[name1](features[name1])
            inferences_0 += torch.sum(name1_embedding * self.embedding_action(torch.zeros_like(features[name1]).long()),
                                      dim=1, keepdim=True)
            inferences_1 += torch.sum(name1_embedding * self.embedding_action(torch.ones_like(features[name1]).long()),
                                      dim=1, keepdim=True)

            if self.args.first_order:
                name1_embedding_first_order = self.embedding_first_order[name1](features[name1])
                inferences_0 += name1_embedding_first_order
                inferences_1 += name1_embedding_first_order
        if self.args.first_order:
            inferences_0 += self.embedding_action_first_order(torch.zeros_like(features[name1]).long())
            inferences_1 += self.embedding_action_first_order(torch.ones_like(features[name1]).long())
        inferences = torch.cat([inferences_0, inferences_1], dim=1)
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        weighted_loss.backward()
        return inferences, weighted_loss, 0

    def genotype(self):
        return "FM", 0

class FM_v2(Virtue_v):
    """
    FM with EE and FC layer
    """
    def __init__(self, embedding_dim, reg, args):
        super(FM_v2, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num, first_order=args.first_order)
        self.args = args
        self._initialize_alphas()
        self.FC = nn.ModuleDict({})
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    self.FC[name1 + ":" + name2] = nn.Linear(embedding_dim, 1, bias=False)
        #initialize contextual infos
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.log_alpha = torch.Tensor([1, 1, 1, 1, 1.0])
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.weights = Variable(torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
        self.rand_array = torch.randn(3000000)

    def reparameterize(self, mu, std):
        std = torch.log(1 + torch.exp(std))
        v = self.rand_array[:std.numel()].reshape(std.shape)
        return (mu + std * v * 0.01)

    def KL_distance(self, mean1, mean2, std1, std2):
        a = torch.log(std2 / std1) + (std1 * std1 + (mean1 - mean2) * (mean1 - mean2)) / 2 / std2 / std2 - 1.0 / 2.0
        return torch.sum(a)

    def recommend(self, features):
        self.eval()
        inferences = 0
        inferences_0 = 0
        inferences_1 = 0
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    name1_embedding_mean = self.embedding_mean[name1](features[name1])
                    name1_embedding_std = self.embedding_std[name1](features[name1])
                    name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                    name2_embedding_mean = self.embedding_mean[name2](features[name2])
                    name2_embedding_std = self.embedding_std[name2](features[name2])
                    name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    inferences += self.FC[name1 + ":" + name2](name1_embedding * name2_embedding)
        inferences_0 = inferences.clone()  # action 0
        inferences_1 = inferences.clone()  # action_1
        #features with action
        for name1 in self.columns:
            name1_embedding_mean = self.embedding_mean[name1](features[name1])
            name1_embedding_std = self.embedding_std[name1](features[name1])
            name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)

            inferences_0 += torch.sum(name1_embedding * self.embedding_action(torch.zeros_like(features[name1]).long()), dim=1, keepdim=True)
            inferences_1 += torch.sum(name1_embedding * self.embedding_action(torch.ones_like(features[name1]).long()), dim=1, keepdim=True)
            if self.args.first_order:
                name1_embedding_first_order = self.embedding_first_order[name1](features[name1])
                inferences_0 += name1_embedding_first_order
                inferences_1 += name1_embedding_first_order
        if self.args.first_order:
            inferences_0 += self.embedding_action_first_order(torch.zeros_like(features[name1]).long())
            inferences_1 += self.embedding_action_first_order(torch.ones_like(features[name1]).long())

        inferences = torch.cat([inferences_0, inferences_1], dim=1)
        pos_weights = torch.zeros_like(features["label"])
        max_index = torch.argmax(inferences, dim=1)
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        self.add_batch(features, pos_weights)
        return reward

    def add_batch(self, features, pos_weights):
        for index in features:
            if index in self.contexts:
                temp = self.contexts[index]
                self.contexts[index] = torch.cat([temp, features[index]], dim=0)
            else:
                self.contexts[index] = features[index]
        self.pos_weights = torch.cat([self.pos_weights, pos_weights], dim=0)

    def step(self, optimizer, arch_optimizer, step):
        self.train()
        losses = []
        train_bandit = utils.get_data_queue_bandit(self.args, self.contexts, self.pos_weights)
        cnt = 0
        for step, features in enumerate(train_bandit):
            cnt += 1
            optimizer.zero_grad()
            output, error_loss, loss_alpha = self.forward(features)
            losses.append(error_loss.cpu().detach().item())
            optimizer.step()
        print("cnt: ", cnt)
        return np.mean(losses)

    def forward(self, features):
        regs = 0
        inferences = 0
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    name1_embedding_mean = self.embedding_mean[name1](features[name1])
                    name1_embedding_std = self.embedding_std[name1](features[name1])
                    name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                    name2_embedding_mean = self.embedding_mean[name2](features[name2])
                    name2_embedding_std = self.embedding_std[name2](features[name2])
                    name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    inferences += self.FC[name1 + ":" + name2](name1_embedding * name2_embedding)

        # features with action
        inferences_0 = inferences.clone()  # action 0
        inferences_1 = inferences.clone()  # action_1
        for name1 in self.columns:
            name1_embedding_mean = self.embedding_mean[name1](features[name1])
            name1_embedding_std = self.embedding_std[name1](features[name1])
            name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
            inferences_0 += torch.sum(name1_embedding * self.embedding_action(torch.zeros_like(features[name1]).long()),
                                      dim=1, keepdim=True)
            inferences_1 += torch.sum(name1_embedding * self.embedding_action(torch.ones_like(features[name1]).long()),
                                      dim=1, keepdim=True)

            if self.args.first_order:
                name1_embedding_first_order = self.embedding_first_order[name1](features[name1])
                inferences_0 += name1_embedding_first_order
                inferences_1 += name1_embedding_first_order
        if self.args.first_order:
            inferences_0 += self.embedding_action_first_order(torch.zeros_like(features[name1]).long())
            inferences_1 += self.embedding_action_first_order(torch.ones_like(features[name1]).long())
        inferences = torch.cat([inferences_0, inferences_1], dim=1)
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        kl = 0
        for name in self.columns:
            kl += self.KL_distance(self.embedding_mean[name].weight,
                                   0 * torch.ones_like(self.embedding_mean[name].weight),
                                   torch.log(1 + torch.exp(self.embedding_std[name].weight)),
                                   0.1 * torch.ones_like(self.embedding_std[name].weight))

        (weighted_loss + kl/features["label"].shape[0]).backward()
        return inferences, weighted_loss, 0

    def genotype(self):
        return "FM_v2", 0

class Random:
    def __init__(self, embedding_dim, reg, args):
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])

    def recommend(self, features):
        pos_weights = torch.zeros_like(features["label"])
        max_index = np.random.randint(0, 2, features["label"].shape[0])
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "Random", 0

class Egreedy:
    def __init__(self, embedding_dim, reg, args):
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        self.epsion = 0.2
        self.action_rewards = {0:[0,1.0], 1:[0.1,1.0]}#total reward, action_num
        self.max_action = 0

    def recommend(self, features):
        max_reward = np.float("-inf")
        for key in self.action_rewards:
            if self.action_rewards[key][0]/self.action_rewards[key][1] > max_reward:
                max_reward = self.action_rewards[key][0]/self.action_rewards[key][1]
                self.max_action = key
        pos_weights = torch.zeros_like(features["label"])
        max_index = np.random.randint(0, 2, features["label"].shape[0])
        simulate_index = []
        for i in range(len(max_index)):
            if np.random.random()<self.epsion:
                simulate_index.append(max_index[i])
            else:
                simulate_index.append(self.max_action)

        a_ind = np.array([(i, val) for i, val in enumerate(simulate_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()

        action_rewards = torch.sum(torch.mul(features["label"], pos_weights), dim=0)
        action_nums = torch.sum(pos_weights, dim=0)
        for key in self.action_rewards:
            temp = self.action_rewards[key]
            temp[0] += action_rewards[key].cpu().detach().item()
            temp[1] += action_nums[key].cpu().detach().item()
            self.action_rewards[key] = temp
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "Egreedy", 0

class Thompson:
    def __init__(self, embedding_dim, reg, args):
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        self.epsion = 0.2
        self.action_rewards = {0:[0,0], 1:[0,0]}#total reward, action_num
        self.max_action = 0

    def recommend(self, features):
        #Thompson sampling
        values = []
        num = 2
        N = 10000
        for index in range(num):
            pos = np.random.beta(1+int(self.action_rewards[index][0]), 2+int(self.action_rewards[index][1]), N)
            values.append(pos)
        action_pos = np.vstack(values)
        action_num = Counter(action_pos.argmax(axis=0))
        action_percentage = []
        for index in range(num):
            action_percentage.append(action_num[index]/N)
        simulate_index = []
        for i in range(features["label"].shape[0]):
            simulate_index.append(np.random.choice(range(num), p=action_percentage))

        pos_weights = torch.zeros_like(features["label"])
        a_ind = np.array([(i, val) for i, val in enumerate(simulate_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()

        action_rewards = torch.sum(torch.mul(features["label"], pos_weights), dim=0)
        action_nums = torch.sum(pos_weights, dim=0)
        for key in self.action_rewards:
            temp = self.action_rewards[key]
            temp[0] += action_rewards[key].cpu().detach().item()
            temp[1] += action_nums[key].cpu().detach().item()
            self.action_rewards[key] = temp
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "Thompson", 0

class LinUCB2:
    def __init__(self, embedding_dim, reg, args):
        self.Aa = torch.eye(104)
        self.ba = torch.zeros(104).view(-1,1)
        self.alpha = 0.05
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])

    def recommend(self, features):
        action1_features = torch.zeros((features["label"].shape[0], 2))
        action1_features[:, 0] = 1.0

        action2_features = torch.zeros((features["label"].shape[0], 2))
        action2_features[:, 1] = 1.0

        action1_input = torch.cat([features["feature"], action1_features], dim=1)
        action2_input = torch.cat([features["feature"], action2_features], dim=1)

        inputs_all = [action1_input, action2_input]
        theta = torch.matmul(torch.inverse(self.Aa), self.ba)

        action1_score = torch.matmul(action1_input, theta) + self.alpha * torch.sqrt(
            torch.sum(torch.mul(torch.matmul(action1_input, torch.inverse(self.Aa)), action1_input), dim=-1)).view(-1,1)
        action2_score = torch.matmul(action2_input, theta) + self.alpha * torch.sqrt(
            torch.sum(torch.mul(torch.matmul(action2_input, torch.inverse(self.Aa)), action2_input), dim=-1)).view(-1, 1)
        score_all = torch.cat([action1_score, action2_score], dim=1)
        max_index = score_all.argmax(dim=1)
        print(Counter(max_index.numpy()))
        pos_weights = torch.zeros_like(features["label"])
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        #update Aa and ba
        for i in range(max_index.shape[0]):
            cur_action = max_index[i].item()
            cur_reward = features["label"][i, cur_action].item()
            cur_feature = inputs_all[cur_action][i]
            self.Aa += torch.matmul(cur_feature.view(-1,1), cur_feature.view(1,-1))
            self.ba += cur_reward * cur_feature.view(-1,1)
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "LinUCB2", 0

#
class LinUCB:
    def __init__(self, embedding_dim, reg, args):
        self.action_num = 2
        self.feature_dim = 102
        self.Aa = []
        self.ba = []
        for i in range(self.action_num):
            self.Aa.append(torch.eye(self.feature_dim))
            self.ba.append(torch.zeros(self.feature_dim).view(-1,1))
        self.alpha = 1.0
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])

    def recommend(self, features):
        score_all = []
        for i in range(self.action_num):
            Aa = self.Aa[i]
            ba = self.ba[i]
            theta = torch.matmul(torch.inverse(Aa), ba)
            score = torch.matmul(features["feature"], theta) + self.alpha * torch.sqrt(
                torch.sum(torch.mul(torch.matmul(features["feature"], torch.inverse(Aa)), features["feature"]), dim=-1)
            ).view(-1,1)
            score_all.append(score)
        score_all = torch.cat(score_all, dim=1)

        max_index = score_all.argmax(dim=1)
        print(Counter(max_index.numpy()))
        pos_weights = torch.zeros_like(features["label"])
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        #update Aa and ba
        for i in range(max_index.shape[0]):
            cur_action = max_index[i].item()
            cur_reward = features["label"][i, cur_action].item()
            cur_feature = features["feature"][i]
            Aa = self.Aa[cur_action]
            ba = self.ba[cur_action]
            Aa += torch.matmul(cur_feature.view(-1,1), cur_feature.view(1,-1))
            ba += cur_reward * cur_feature.view(-1,1)
            self.Aa[cur_action] = Aa
            self.ba[cur_action] = ba
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "LinUCB", 0

class LinThompson:
    def __init__(self, embedding_dim, reg, args):
        self.action_num = 2
        self.feature_dim = 102
        self.Aa = []
        self.ba = []
        for i in range(self.action_num):
            self.Aa.append(torch.eye(self.feature_dim))
            self.ba.append(torch.zeros(self.feature_dim).view(-1, 1))
        self.alpha = 1.0
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])

    def recommend(self, features):
        score_all = []
        for i in range(self.action_num):
            Aa = self.Aa[i]
            ba = self.ba[i]
            mu = torch.matmul(torch.inverse(Aa), ba)
            variance = torch.inverse(Aa)
            try:
                theta = MultivariateNormal(loc=mu.view(-1), covariance_matrix=self.alpha * variance).sample().view(-1,1)
            except:
                print("Error here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                theta = mu.view(-1,1)
            score = torch.matmul(features["feature"], theta) + self.alpha * torch.sqrt(
                torch.sum(torch.mul(torch.matmul(features["feature"], torch.inverse(Aa)), features["feature"]), dim=-1)
            ).view(-1, 1)
            score_all.append(score)
        score_all = torch.cat(score_all, dim=1)

        max_index = score_all.argmax(dim=1)
        print(Counter(max_index.numpy()))
        pos_weights = torch.zeros_like(features["label"])
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        # update Aa and ba
        for i in range(max_index.shape[0]):
            cur_action = max_index[i].item()
            cur_reward = features["label"][i, cur_action].item()
            cur_feature = features["feature"][i]
            Aa = self.Aa[cur_action]
            ba = self.ba[cur_action]
            Aa += torch.matmul(cur_feature.view(-1, 1), cur_feature.view(1, -1))
            ba += cur_reward * cur_feature.view(-1, 1)
            self.Aa[cur_action] = Aa
            self.ba[cur_action] = ba
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "LinThompson", 0

class LinEGreedy:
    def __init__(self, embedding_dim, reg, args):
        self.Aa = torch.eye(104)
        self.ba = torch.zeros(104).view(-1,1)
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        self.epsion = 0.2
        self.turn = True

    def recommend(self, features):
        action1_features = torch.zeros((features["label"].shape[0], 2))
        action1_features[:, 0] = 1.0

        action2_features = torch.zeros((features["label"].shape[0], 2))
        action2_features[:, 1] = 1.0

        action1_input = torch.cat([features["feature"], action1_features], dim=1)
        action2_input = torch.cat([features["feature"], action2_features], dim=1)

        inputs_all = [action1_input, action2_input]
        theta = torch.matmul(torch.inverse(self.Aa), self.ba)

        action1_score = torch.matmul(action1_input, theta)
        action2_score = torch.matmul(action2_input, theta)
        score_all = torch.cat([action1_score, action2_score], dim=1)
        max_index = score_all.argmax(dim=1)
        if self.turn:
            simulate_index = []
            for i in range(len(max_index)):
                if np.random.random() < self.epsion:
                    simulate_index.append(max_index[i].item())
                else:
                    simulate_index.append(np.random.randint(0, 2))
            max_index = simulate_index
            self.turn = False
        print(Counter(max_index))
        pos_weights = torch.zeros_like(features["label"])
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        #update Aa and ba
        for i in range(len(max_index)):
            cur_action = max_index[i]
            cur_reward = features["label"][i, cur_action].item()
            cur_feature = inputs_all[cur_action][i]
            self.Aa += torch.matmul(cur_feature.view(-1,1), cur_feature.view(1,-1))
            self.ba += cur_reward * cur_feature.view(-1,1)
        return reward

    def step(self, optimizer, arch_optimizer, epoch):
        return 0

    def genotype(self):
        return "LinEGreedy", 0