import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import utils
import ipdb

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

def constrain(p):
    c = torch.norm(p, p=2, dim=1, keepdim=True)
    c[c < 1] = 1.0
    p.data.div_(c)

def MixedBinary(embedding_p, embedding_q, weights, FC):
    # return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
    #                               for w, primitive, fc in zip(weights, PRIMITIVES_BINARY, FC)]), 0)
    pos = weights.argmax().item()
    return weights[pos] * FC[pos](OPS[PRIMITIVES_BINARY[pos]](embedding_p, embedding_q))

class Virtue(nn.Module):

    def __init__(self, embedding_dim, reg, embedding_num=12, ofm=False):
        super(Virtue, self).__init__()
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.embedding_mean = nn.ModuleDict({})
        self.embedding_std = nn.ModuleDict({})
        #self.embedding_first_order = nn.ModuleDict({})
        self.columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

        # for name in self.columns:
        #     self.embedding_first_order[name] = nn.Embedding(embedding_num, 2)

        if not ofm:
            for name in self.columns:
                self.embedding_mean[name] = nn.Embedding(embedding_num, embedding_dim)
                self.embedding_std[name] = nn.Embedding(embedding_num, embedding_dim)
        else:
            for name in self.columns:
                temp_mean = nn.ModuleList()
                temp_std = nn.ModuleList()
                for primitive in PRIMITIVES_BINARY:
                    temp_mean.append(nn.Embedding(embedding_num, embedding_dim))
                    temp_std.append(nn.Embedding(embedding_num, embedding_dim))
                self.embedding_mean[name] = temp_mean
                self.embedding_std[name] = temp_std

class Virtue2(nn.Module):

    def __init__(self, embedding_dim, reg, embedding_num=12, ofm=False):
        super(Virtue2, self).__init__()
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.embedding_all = nn.ModuleDict({})
        self.columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

        if not ofm:
            for name in self.columns:
                self.embedding_all[name] = nn.Embedding(embedding_num, embedding_dim)
        else:
            for name in self.columns:
                temp = nn.ModuleList()
                for primitive in PRIMITIVES_BINARY:
                    temp.append(nn.Embedding(embedding_num, embedding_dim))
                self.embedding_all[name] = temp

class DSNAS_v(Virtue):

    def __init__(self, embedding_dim, reg, args):
        super(DSNAS_v, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self.FC = nn.ModuleDict({})
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    temp = nn.ModuleList()
                    for primitive in PRIMITIVES_BINARY:
                        if primitive == 'concat':
                            temp.append(nn.Linear(2*embedding_dim, 2, bias=False))
                        else:
                            temp.append(nn.Linear(embedding_dim, 2, bias=False))
                    self.FC[name1 + ":" + name2] = temp
        self.args = args
        self._initialize_alphas()
        #initialize contextual infos
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        if self.args.multi_operation:
            num_op = len(self.columns)
            #self.log_alpha = torch.nn.Parameter(torch.zeros((int(num_op*(num_op-1)/2), len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
            self.log_alpha = torch.nn.Parameter(torch.tensor([[1.1703, -0.2041, 0.2789, 1.6233, 1.2953],
                    [1.7939, -1.9228, 1.4970, 0.1009, 1.2945],
                    [0.9141, 0.0278, 1.0914, 1.4635, 0.9335],
                    [-0.8395, -0.6790, 1.6929, 1.3102, 1.3892],
                    [0.6935, -1.1365, 1.5669, 1.2360, 1.0528],
                    [0.8841, -0.1637, 1.4551, 1.0775, 1.0721],
                    [1.2669, -0.5721, 0.5361, 1.5923, 1.0500],
                    [1.0271, 0.3320, 0.9572, 1.0095, 1.4277],
                    [1.0657, 0.2153, 0.5999, 1.1211, 1.4865],
                    [1.6462, -0.6197, 1.6147, 1.9634, 0.3100],
                    [1.4413, 0.0692, 0.9746, 1.0370, 1.0501],
                    [1.4896, -0.0137, 0.8945, 0.9403, 1.1247],
                    [1.4569, 0.3129, 1.0348, 0.9539, 1.0619],
                    [0.6010, -1.0310, 1.2261, 1.5945, 1.2568],
                    [1.6003, -0.9270, 1.1398, 0.4440, 1.2788],
                    [1.1263, -1.0382, 0.9785, 1.4928, 1.1206],
                    [1.5846, -0.2864, 1.2341, 0.8667, 1.0676],
                    [1.2558, -1.5168, 1.1708, 1.5934, 0.7656],
                    [0.4929, -0.5358, 1.2603, 1.0061, 1.5858],
                    [1.0267, -0.2059, 0.9694, 1.0517, 1.4342],
                    [0.9599, -0.7199, 1.5792, 0.9892, 1.2341],
                    [0.9901, 0.1213, 0.4225, 1.2474, 1.5794],
                    [0.8617, -3.4300, 2.3946, 2.6207, -0.9823],
                    [1.2378, -2.6564, 0.8267, 1.2131, 1.5809],
                    [0.4944, -0.7097, 1.4089, 1.7146, 1.0155],
                    [0.8259, -0.3022, 1.5460, 1.1121, 1.1885],
                    [0.7024, 0.0810, 1.6000, 1.2560, 1.0186],
                    [0.9505, -0.8578, 1.0379, 1.5584, 1.2087]]).requires_grad_())

        else:
            self.log_alpha = torch.nn.Parameter(torch.zeros((1, len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        self._arch_parameters = [self.log_alpha]
        self.weights = Variable(torch.zeros_like(self.log_alpha))
        if self.args.early_fix_arch:
            self.fix_arch_index = {}
        self.rand_array = torch.randn(3000000)

    def reparameterize(self, mu, std):
        std = torch.log(1 + torch.exp(std))
        #v = self.rand_array[:std.numel()].reshape(std.shape)
        v = torch.randn(3000000)[:std.numel()].reshape(std.shape)
        return (mu + std * v * 0.01)

    # def KL_distance(self, mean1, mean2, std1, std2):
    #     a = torch.log(std2 / std1) + (std1 * std1 + (mean1 - mean2) * (mean1 - mean2)) / 2 / std2 / std2 - 1.0 / 2.0
    #     return torch.sum(a)

    def KL_distance(self, mean1, mean2, std1, std2):
        a = 1/2 * (torch.log(torch.det(std2)/torch.det(std1)) - std1.numel() + (mean1 - mean2)*(mean1 - mean2)/torch.square(std2) +
               torch.sum(torch.square(std1)/torch.square(std2)))
        return a

    def recommend(self, features):
        self.eval()
        self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim=-1).view(-1, 1), 1)
        if self.args.early_fix_arch:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.weights[key, :].zero_()
                    self.weights[key, value_lst[0]] = 1
        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        name1_embedding_mean = self.embedding_mean[name1][max_index](features[name1])
                        name1_embedding_std = self.embedding_std[name1][max_index](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2][max_index](features[name2])
                        name2_embedding_std = self.embedding_std[name2][max_index](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    else:
                        name1_embedding_mean = self.embedding_mean[name1](features[name1])
                        name1_embedding_std = self.embedding_std[name1](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2](features[name2])
                        name2_embedding_std = self.embedding_std[name2](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    if self.args.trans:
                        name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                        name2_embedding_trans = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    else:
                        name1_embedding_trans = name1_embedding
                        name2_embedding_trans = name2_embedding
                    inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        if self.args.first_order:
            for name in self.columns:
                inferences += self.embedding_first_order[name](features[name])

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
        #if epoch < self.args.search_epoch:
        #    train_epoch = (epoch+1)*5
        #else:
        train_epoch = 5
        for k in range(train_epoch):
            for step, features in enumerate(train_bandit):
                optimizer.zero_grad()
                if epoch < self.args.search_epoch:
                    arch_optimizer.zero_grad()
                    output, error_loss, loss_alpha = self.forward(features, epoch, search=True)
                else:
                    output, error_loss, loss_alpha = self.forward(features, epoch, search=False)
                losses.append(error_loss.cpu().detach().item())
                optimizer.step()
                if epoch < self.args.search_epoch:
                    arch_optimizer.step()
        return np.mean(losses)

    def revised_arch_index(self, epoch):
        if self.args.early_fix_arch:
            if epoch < self.args.search_epoch:
                sort_log_alpha = torch.topk(F.softmax(self.log_alpha.data, dim=-1), 2)
                argmax_index = (sort_log_alpha[0][:, 0] - sort_log_alpha[0][:, 1] >= 0.10)
                for id in range(argmax_index.size(0)):
                    if argmax_index[id] == 1 and id not in self.fix_arch_index.keys():
                        self.fix_arch_index[id] = [sort_log_alpha[1][id, 0].item(),
                                                    self.log_alpha.detach().clone()[id, :]]
            if epoch >= self.args.search_epoch:
                #fix the arch。
                max_index = torch.argmax(self.log_alpha, dim=-1)
                for id in range(max_index.size(0)):
                    if id not in self.fix_arch_index.keys():
                        self.fix_arch_index[id] = [max_index[id].item(), self.log_alpha.detach().clone()[id, :]]

            for key, value_lst in self.fix_arch_index.items():
                self.log_alpha.data[key, :] = value_lst[1]

    def forward(self, features, epoch, search):
        #self.weights = self._get_weights(self.log_alpha)
        self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim=-1).view(-1, 1), 1)
        # self.revised_arch_index(epoch)
        # if self.args.early_fix_arch:
        #     if len(self.fix_arch_index.keys()) > 0:
        #         for key, value_lst in self.fix_arch_index.items():
        #             self.weights[key, :].zero_()
        #             self.weights[key, value_lst[0]] = 1
        if search:
            cate_prob = F.softmax(self.log_alpha, dim=-1)
            self.cate_prob = cate_prob.clone().detach()
            loss_alpha = torch.log(
                (self.weights * F.softmax(self.log_alpha, dim=-1)).sum(-1)).sum()
            self.weights.requires_grad_()

        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        from sklearn.externals.joblib import Parallel, delayed
        names_all = []
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        name1_embedding_mean = self.embedding_mean[name1][max_index](features[name1])
                        name1_embedding_std = self.embedding_std[name1][max_index](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2][max_index](features[name2])
                        name2_embedding_std = self.embedding_std[name2][max_index](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    else:
                        name1_embedding_mean = self.embedding_mean[name1](features[name1])
                        name1_embedding_std = self.embedding_std[name1](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2](features[name2])
                        name2_embedding_std = self.embedding_std[name2](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    if self.args.trans:
                        name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                        name2_embedding_trans = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    else:
                        name1_embedding_trans = name1_embedding
                        name2_embedding_trans = name2_embedding
                    names_all.append(
                        [name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1, ), self.FC[name1 + ":" + name2]])
        res = Parallel(n_jobs=8, backend="threading")(
            delayed(MixedBinary)(para1, para2, para3, para4) for para1, para2, para3, para4 in names_all)
        inferences = sum(res)
        if self.args.first_order:
            for name in self.columns:
                inferences += self.embedding_first_order[name](features[name])
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        kl = 0

        for name in self.columns:
            if not self.args.ofm:
                kl += self.KL_distance(self.embedding_mean[name].weight,
                                       0 * torch.ones_like(self.embedding_mean[name].weight),
                                       torch.log(1 + torch.exp(self.embedding_std[name].weight)),
                                       0.1 * torch.ones_like(self.embedding_std[name].weight))
            else:
                for index in range(len(PRIMITIVES_BINARY)):
                    kl += self.KL_distance(self.embedding_mean[name][index].weight,
                                       0 * torch.ones_like(self.embedding_mean[name][index].weight),
                                       torch.log(1 + torch.exp(self.embedding_std[name][index].weight)),
                                       0.1 * torch.ones_like(self.embedding_std[name][index].weight))
        if search:
            self.weights.grad = torch.zeros_like(self.weights)
            (weighted_loss + loss_alpha + kl/features["label"].shape[0]).backward()
            self.block_reward = self.weights.grad.data.sum(-1)
            self.log_alpha.grad.data.mul_(self.block_reward.view(-1, 1))
            return inferences, weighted_loss, loss_alpha
        else:
            (weighted_loss + kl/features["label"].shape[0]).backward()
            return inferences, weighted_loss, 0

    def _get_weights(self, log_alpha):
        if self.args.random_sample:
            uni = torch.ones_like(log_alpha)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
        else:
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        if not self.args.multi_operation:
            genotype = PRIMITIVES_BINARY[self.log_alpha.argmax().cpu().numpy()]
            genotype_p = F.softmax(self.log_alpha, dim=-1)
        else:
            genotype = []
            for index in self.log_alpha.argmax(axis=1).cpu().numpy():
                genotype.append(PRIMITIVES_BINARY[index])
            genotype = ":".join(genotype[:10])
            genotype_p = F.softmax(self.log_alpha, dim=-1)[:10]
        return genotype, genotype_p.cpu().detach()

class NASP(Virtue2):

    def __init__(self, embedding_dim, reg, args):
        super(NASP, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self.FC = nn.ModuleDict({})
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    temp = nn.ModuleList()
                    for primitive in PRIMITIVES_BINARY:
                        if primitive == 'concat':
                            temp.append(nn.Linear(2*embedding_dim, 2, bias=False))
                        else:
                            temp.append(nn.Linear(embedding_dim, 2, bias=False))
                    self.FC[name1 + ":" + name2] = temp
        self.args = args
        self._initialize_alphas()
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        if self.args.multi_operation:
            num_op = len(self.columns)
            self.log_alpha = torch.nn.Parameter(torch.zeros((int(num_op*(num_op-1)/2), len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        else:
            self.log_alpha = torch.nn.Parameter(torch.zeros((1, len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        self._arch_parameters = [self.log_alpha]
        self.weights = Variable(torch.zeros_like(self.log_alpha))
        if self.args.early_fix_arch:
            self.fix_arch_index = {}
        self.rand_array = torch.randn(3000000)

    def recommend(self, features):
        self.eval()
        self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim=-1).view(-1, 1), 1)
        if self.args.early_fix_arch:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.weights[key, :].zero_()
                    self.weights[key, value_lst[0]] = 1
        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        name1_embedding = self.embedding_all[name1][max_index](features[name1])
                        name2_embedding = self.embedding_all[name2][max_index](features[name2])
                    else:
                        name1_embedding = self.embedding_all[name1](features[name1])
                        name2_embedding = self.embedding_all[name2](features[name2])
                    if self.args.trans:
                        name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                        name2_embedding_trans = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    else:
                        name1_embedding_trans = name1_embedding
                        name2_embedding_trans = name2_embedding
                    inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        if self.args.first_order:
            for name in self.columns:
                inferences += self.embedding_first_order[name](features[name])

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

    def MixedBinary_ofm(self, embedding_p_all, embedding_q_all, weights, FC):
        return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
                                      for w, primitive, fc, embedding_p, embedding_q in zip(weights, PRIMITIVES_BINARY, FC, embedding_p_all, embedding_q_all)]), 0)

    def MixedBinary_all(self, embedding_p, embedding_q, weights, FC):
        return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
                                      for w, primitive, fc in zip(weights, PRIMITIVES_BINARY, FC)]), 0)

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
                arch_optimizer.zero_grad()
                output, error_loss, loss_alpha = self.forward(features, epoch, search=True)
                losses.append(error_loss.cpu().detach().item())
                optimizer.step()
                arch_optimizer.step()
        return np.mean(losses)

    def forward(self, features, epoch, search):
        self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim=-1).view(-1, 1), 1)
        self.weights.requires_grad_()

        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        from sklearn.externals.joblib import Parallel, delayed
        names_all = []
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        embedding_name1_all = []
                        embedding_name2_all = []
                        for index_name in range(len(PRIMITIVES_BINARY)):
                            name1_embedding = self.embedding_all[name1][index_name](features[name1])
                            embedding_name1_all.append(name1_embedding)
                            name2_embedding = self.embedding_all[name2][index_name](features[name2])
                            embedding_name2_all.append(name2_embedding)
                    else:
                        name1_embedding = self.embedding_all[name1](features[name1])
                        name2_embedding = self.embedding_all[name2](features[name2])
                    if self.args.trans:
                        if self.args.ofm:
                            embedding_name1_all_temp = []
                            embedding_name2_all_temp = []
                            for index_temp in range(len(embedding_name1_all)):
                                embedding_name1_all_temp.append(self.mlp_p(embedding_name1_all[index_temp].view(-1, 1)).view(embedding_name1_all[index_temp].size()))
                                embedding_name2_all_temp.append(self.mlp_p(embedding_name2_all[index_temp].view(-1, 1)).view(embedding_name2_all[index_temp].size()))
                            embedding_name1_all = embedding_name1_all_temp
                            embedding_name2_all = embedding_name2_all_temp
                        else:
                            name1_embedding = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                            name2_embedding = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    if self.args.ofm:
                        names_all.append([embedding_name1_all, embedding_name2_all, cur_weights.view(-1, ), self.FC[name1 + ":" + name2]])
                    else:
                        names_all.append(
                            [name1_embedding, name2_embedding, cur_weights.view(-1, ), self.FC[name1 + ":" + name2]])
        if self.args.ofm:
            res = Parallel(n_jobs=8, backend="threading")(
                delayed(self.MixedBinary_ofm)(para1, para2, para3, para4) for para1, para2, para3, para4 in names_all)
        else:
            res = Parallel(n_jobs=8, backend="threading")(
                delayed(self.MixedBinary_all)(para1, para2, para3, para4) for para1, para2, para3, para4 in names_all)
        inferences = sum(res)
        if self.args.first_order:
            for name in self.columns:
                inferences += self.embedding_first_order[name](features[name])
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        weighted_loss.backward()
        self.log_alpha.grad = self.weights.grad
        return inferences, weighted_loss, 0

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        if not self.args.multi_operation:
            genotype = PRIMITIVES_BINARY[self.log_alpha.argmax().cpu().numpy()]
            genotype_p = F.softmax(self.log_alpha, dim=-1)
        else:
            genotype = []
            for index in self.log_alpha.argmax(axis=1).cpu().numpy():
                genotype.append(PRIMITIVES_BINARY[index])
            genotype = ":".join(genotype[:10])
            genotype_p = F.softmax(self.log_alpha, dim=-1)[:10]
        return genotype, genotype_p.cpu().detach()


class NASP_v(Virtue):

    def __init__(self, embedding_dim, reg, args):
        super(NASP_v, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self.FC = nn.ModuleDict({})
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    temp = nn.ModuleList()
                    for primitive in PRIMITIVES_BINARY:
                        if primitive == 'concat':
                            temp.append(nn.Linear(2*embedding_dim, 2, bias=False))
                        else:
                            temp.append(nn.Linear(embedding_dim, 2, bias=False))
                    self.FC[name1 + ":" + name2] = temp
        self.args = args
        self._initialize_alphas()
        #initialize contextual infos
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        if self.args.multi_operation:
            num_op = len(self.columns)
            self.log_alpha = torch.nn.Parameter(torch.zeros((int(num_op*(num_op-1)/2), len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        else:
            self.log_alpha = torch.nn.Parameter(torch.zeros((1, len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        self._arch_parameters = [self.log_alpha]
        self.weights = Variable(torch.zeros_like(self.log_alpha))
        if self.args.early_fix_arch:
            self.fix_arch_index = {}
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
        self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim=-1).view(-1, 1), 1)
        if self.args.early_fix_arch:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.weights[key, :].zero_()
                    self.weights[key, value_lst[0]] = 1
        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        name1_embedding_mean = self.embedding_mean[name1][max_index](features[name1])
                        name1_embedding_std = self.embedding_std[name1][max_index](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2][max_index](features[name2])
                        name2_embedding_std = self.embedding_std[name2][max_index](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    else:
                        name1_embedding_mean = self.embedding_mean[name1](features[name1])
                        name1_embedding_std = self.embedding_std[name1](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2](features[name2])
                        name2_embedding_std = self.embedding_std[name2](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    if self.args.trans:
                        name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                        name2_embedding_trans = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    else:
                        name1_embedding_trans = name1_embedding
                        name2_embedding_trans = name2_embedding
                    inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        if self.args.first_order:
            for name in self.columns:
                inferences += self.embedding_first_order[name](features[name])

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

    def MixedBinary_ofm(self, embedding_p_all, embedding_q_all, weights, FC):
        return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
                                      for w, primitive, fc, embedding_p, embedding_q in zip(weights, PRIMITIVES_BINARY, FC, embedding_p_all, embedding_q_all)]), 0)

    def MixedBinary_all(self, embedding_p, embedding_q, weights, FC):
        return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
                                      for w, primitive, fc in zip(weights, PRIMITIVES_BINARY, FC)]), 0)

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
                arch_optimizer.zero_grad()
                output, error_loss, loss_alpha = self.forward(features, epoch, search=True)
                # if epoch < self.args.search_epoch:
                #     arch_optimizer.zero_grad()
                #     output, error_loss, loss_alpha = self.forward(features, epoch, search=True)
                # else:
                #     output, error_loss, loss_alpha = self.forward(features, epoch, search=False)
                losses.append(error_loss.cpu().detach().item())
                optimizer.step()
                arch_optimizer.step()
                # if epoch < self.args.search_epoch:
                #     arch_optimizer.step()
        return np.mean(losses)

    def revised_arch_index(self, epoch):
        if self.args.early_fix_arch:
            if epoch < self.args.search_epoch:
                sort_log_alpha = torch.topk(F.softmax(self.log_alpha.data, dim=-1), 2)
                argmax_index = (sort_log_alpha[0][:, 0] - sort_log_alpha[0][:, 1] >= 0.10)
                for id in range(argmax_index.size(0)):
                    if argmax_index[id] == 1 and id not in self.fix_arch_index.keys():
                        self.fix_arch_index[id] = [sort_log_alpha[1][id, 0].item(),
                                                    self.log_alpha.detach().clone()[id, :]]
            if epoch >= self.args.search_epoch:
                #fix the arch。
                max_index = torch.argmax(self.log_alpha, dim=-1)
                for id in range(max_index.size(0)):
                    if id not in self.fix_arch_index.keys():
                        self.fix_arch_index[id] = [max_index[id].item(), self.log_alpha.detach().clone()[id, :]]

            for key, value_lst in self.fix_arch_index.items():
                self.log_alpha.data[key, :] = value_lst[1]

    def forward(self, features, epoch, search):
        self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim=-1).view(-1, 1), 1)
        self.weights.requires_grad_()
        regs = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        from sklearn.externals.joblib import Parallel, delayed
        names_all = []
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        embedding_name1_all = []
                        embedding_name2_all = []
                        for index_name in range(len(PRIMITIVES_BINARY)):
                            name1_embedding_mean = self.embedding_mean[name1][index_name](features[name1])
                            name1_embedding_std = self.embedding_std[name1][index_name](features[name1])
                            name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                            embedding_name1_all.append(name1_embedding)

                            name2_embedding_mean = self.embedding_mean[name2][index_name](features[name2])
                            name2_embedding_std = self.embedding_std[name2][index_name](features[name2])
                            name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                            embedding_name2_all.append(name2_embedding)
                    else:
                        name1_embedding_mean = self.embedding_mean[name1](features[name1])
                        name1_embedding_std = self.embedding_std[name1](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2](features[name2])
                        name2_embedding_std = self.embedding_std[name2](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    regs += 1e-5 * (torch.norm(name1_embedding) + torch.norm(name2_embedding))
                    if self.args.trans:
                        if self.args.ofm:
                            embedding_name1_all_temp = []
                            embedding_name2_all_temp = []
                            for index_temp in range(len(embedding_name1_all)):
                                embedding_name1_all_temp.append(self.mlp_p(embedding_name1_all[index_temp].view(-1, 1)).view(embedding_name1_all[index_temp].size()))
                                embedding_name2_all_temp.append(self.mlp_p(embedding_name2_all[index_temp].view(-1, 1)).view(embedding_name2_all[index_temp].size()))
                            embedding_name1_all = embedding_name1_all_temp
                            embedding_name2_all = embedding_name2_all_temp
                        else:
                            name1_embedding = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                            name2_embedding = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    if self.args.ofm:
                        names_all.append([embedding_name1_all, embedding_name2_all, cur_weights.view(-1, ), self.FC[name1 + ":" + name2]])
                    else:
                        names_all.append(
                            [name1_embedding, name2_embedding, cur_weights.view(-1, ), self.FC[name1 + ":" + name2]])
        if self.args.ofm:
            res = Parallel(n_jobs=8, backend="threading")(
                delayed(self.MixedBinary_ofm)(para1, para2, para3, para4) for para1, para2, para3, para4 in names_all)
        else:
            res = Parallel(n_jobs=8, backend="threading")(
                delayed(self.MixedBinary_all)(para1, para2, para3, para4) for para1, para2, para3, para4 in names_all)
        inferences = sum(res)
        if self.args.first_order:
            for name in self.columns:
                inferences += self.embedding_first_order[name](features[name])
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        kl = 0

        for name in self.columns:
            if not self.args.ofm:
                kl += self.KL_distance(self.embedding_mean[name].weight,
                                       0 * torch.ones_like(self.embedding_mean[name].weight),
                                       torch.log(1 + torch.exp(self.embedding_std[name].weight)),
                                       0.1 * torch.ones_like(self.embedding_std[name].weight))
            else:
                for index in range(len(PRIMITIVES_BINARY)):
                    kl += self.KL_distance(self.embedding_mean[name][index].weight,
                                       0 * torch.ones_like(self.embedding_mean[name][index].weight),
                                       torch.log(1 + torch.exp(self.embedding_std[name][index].weight)),
                                       0.1 * torch.ones_like(self.embedding_std[name][index].weight))

        (weighted_loss + regs + kl/features["label"].shape[0]).backward()
        self.log_alpha.grad = self.weights.grad
        return inferences, weighted_loss, 0

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        if not self.args.multi_operation:
            genotype = PRIMITIVES_BINARY[self.log_alpha.argmax().cpu().numpy()]
            genotype_p = F.softmax(self.log_alpha, dim=-1)
        else:
            genotype = []
            for index in self.log_alpha.argmax(axis=1).cpu().numpy():
                genotype.append(PRIMITIVES_BINARY[index])
            genotype = ":".join(genotype[:10])
            genotype_p = F.softmax(self.log_alpha, dim=-1)[:10]
        return genotype, genotype_p.cpu().detach()

class MULTIPLY_v(Virtue):

    def __init__(self, embedding_dim, reg, args):
        super(MULTIPLY_v, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self.FC = nn.ModuleDict({})
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    temp = nn.ModuleList()
                    for primitive in PRIMITIVES_BINARY:
                        if primitive == 'concat':
                            temp.append(nn.Linear(2 * embedding_dim, 2, bias=False))
                        else:
                            temp.append(nn.Linear(embedding_dim, 2, bias=False))
                    self.FC[name1 + ":" + name2] = temp

        self.args = args
        self._initialize_alphas()
        #initialize contextual infos
        self.contexts = {}
        self.pos_weights = torch.Tensor()
        self.rewards = []

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.log_alpha = torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0])
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
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        name1_embedding_mean = self.embedding_mean[name1][max_index](features[name1])
                        name1_embedding_std = self.embedding_std[name1][max_index](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2][max_index](features[name2])
                        name2_embedding_std = self.embedding_std[name2][max_index](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    else:
                        name1_embedding_mean = self.embedding_mean[name1](features[name1])
                        name1_embedding_std = self.embedding_std[name1](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2](features[name2])
                        name2_embedding_std = self.embedding_std[name2](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    if self.args.trans:
                        name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                        name2_embedding_trans = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    else:
                        name1_embedding_trans = name1_embedding
                        name2_embedding_trans = name2_embedding
                    inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        # if self.args.first_order:
        #     for name in self.columns:
        #         inferences += self.embedding_first_order[name](features[name])
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
        if epoch < 10:
            train_epoch = (epoch+1)*5
        else:
            train_epoch = 1
        for i in range(train_epoch):
            for step, features in enumerate(train_bandit):
                optimizer.zero_grad()
                output, error_loss, loss_alpha = self.forward(features)
                losses.append(error_loss.cpu().detach().item())
                optimizer.step()
        return np.mean(losses)

    def forward(self, features):
        regs = 0
        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        from sklearn.externals.joblib import Parallel, delayed
        names_all = []
        for index1, name1 in enumerate(self.columns):
            for index2, name2 in enumerate(self.columns):
                if index1 < index2:
                    if self.args.multi_operation:
                        cur_weights = self.weights[cur_index]
                        max_index = cur_weights.argmax().item()
                        cur_index += 1
                    if self.args.ofm:
                        name1_embedding_mean = self.embedding_mean[name1][max_index](features[name1])
                        name1_embedding_std = self.embedding_std[name1][max_index](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2][max_index](features[name2])
                        name2_embedding_std = self.embedding_std[name2][max_index](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    else:
                        name1_embedding_mean = self.embedding_mean[name1](features[name1])
                        name1_embedding_std = self.embedding_std[name1](features[name1])
                        name1_embedding = self.reparameterize(name1_embedding_mean, name1_embedding_std)
                        name2_embedding_mean = self.embedding_mean[name2](features[name2])
                        name2_embedding_std = self.embedding_std[name2](features[name2])
                        name2_embedding = self.reparameterize(name2_embedding_mean, name2_embedding_std)
                    if self.args.trans:
                        name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                        name2_embedding_trans = self.mlp_q(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                    else:
                        name1_embedding_trans = name1_embedding
                        name2_embedding_trans = name2_embedding
                    names_all.append(
                        [name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1, ), self.FC[name1 + ":" + name2]])
        res = Parallel(n_jobs=8, backend="threading")(
            delayed(MixedBinary)(para1, para2, para3, para4) for para1, para2, para3, para4 in names_all)
        inferences = sum(res)
        # if self.args.first_order:
        #     for name in self.columns:
        #         inferences += self.embedding_first_order[name](features[name])
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        kl = 0
        for name in self.columns:
            if not self.args.ofm:
                kl += self.KL_distance(self.embedding_mean[name].weight,
                                       0 * torch.ones_like(self.embedding_mean[name].weight),
                                       torch.log(1 + torch.exp(self.embedding_std[name].weight)),
                                       0.1 * torch.ones_like(self.embedding_std[name].weight))
            else:
                kl += self.KL_distance(self.embedding_mean[name].weight,
                                       0 * torch.ones_like(self.embedding_mean[name].weight),
                                       torch.log(1 + torch.exp(self.embedding_std[name].weight)),
                                       0.1 * torch.ones_like(self.embedding_std[name].weight))

        (weighted_loss + kl/features["label"].shape[0]).backward()
        return inferences, weighted_loss, 0

    def genotype(self):
        return "FM_v", 0

class MAX_v(MULTIPLY_v):
    def __init__(self, embedding_dim, reg, args):
        super(MAX_v, self).__init__(embedding_dim, reg, args)

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.log_alpha = torch.Tensor([1, 1, 1, 1, 1.0])
        self.weights = Variable(torch.Tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
        self.rand_array = torch.randn(3000000)

    def genotype(self):
        return "MAX", 0

class PLUS_v(MULTIPLY_v):
    def __init__(self, embedding_dim, reg, args):
        super(PLUS_v, self).__init__(embedding_dim, reg, args)

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.log_alpha = torch.Tensor([1, 0, 0, 0, 0.0])
        self.weights = Variable(torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
        self.rand_array = torch.randn(3000000)

    def genotype(self):
        return "PLUS", 0

class MIN_v(MULTIPLY_v):
    def __init__(self, embedding_dim, reg, args):
        super(MIN_v, self).__init__(embedding_dim, reg, args)

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.log_alpha = torch.Tensor([0, 0, 0, 1, 0.0])
        self.weights = Variable(torch.Tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
        self.rand_array = torch.randn(3000000)

    def genotype(self):
        return "MIN", 0

class CONCAT_v(MULTIPLY_v):
    def __init__(self, embedding_dim, reg, args):
        super(CONCAT_v, self).__init__(embedding_dim, reg, args)

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.log_alpha = torch.Tensor([0, 0, 0, 0, 1.0])
        self.weights = Variable(torch.Tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
        self.rand_array = torch.randn(3000000)

    def genotype(self):
        return "CONCAT", 0
