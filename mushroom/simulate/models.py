import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import utils
import time

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
    return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
                                  for w,primitive,fc in zip(weights,PRIMITIVES_BINARY,FC)]), 0)

class Virtue(nn.Module):

    def __init__(self, embedding_dim, reg, embedding_num=12, ofm=False):
        super(Virtue, self).__init__()
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.embedding_all = nn.ModuleDict({})
        self.columns = ["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
        if not ofm:
            for name in self.columns:
                self.embedding_all[name] = nn.Embedding(embedding_num, embedding_dim)
        else:
            for name in self.columns:
                temp = nn.ModuleList()
                for primitive in PRIMITIVES_BINARY:
                    temp.append(nn.Embedding(embedding_num, embedding_dim))
                self.embedding_all[name] = temp

    def compute_loss(self, inferences, labels, regs):
        labels = torch.reshape(labels, [-1,1])
        loss = F.binary_cross_entropy_with_logits(inferences, labels.float())
        #loss = F.mse_loss(inferences, labels)
        return loss + regs

class Network(Virtue):

    def __init__(self, embedding_dim, arch, reg):
        super(Network, self).__init__(embedding_dim, reg)
        self.arch = arch
        self.mlp_p = arch['mlp']['p']
        self.mlp_q = arch['mlp']['q']

        self.FC = nn.ModuleDict({})
        for name1 in self.columns:
            for name2 in self.columns:
                if arch['binary'] == 'concat':
                    self.FC[name1+ ":" + name2] = nn.Linear(2*embedding_dim, 1, bias=False)
                else:
                    self.FC[name1 + ":" + name2] = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, features):
        for value in self.FC.values():
            constrain(next(value.parameters()))
        inferences = 0
        regs = 0
        for name1 in self.columns:
            for name2 in self.columns:
                name1_embedding = self.embedding_all[name1](features[name1])
                name2_embedding = self.embedding_all[name2](features[name2])
                regs += self.reg * (torch.norm(name1_embedding) + torch.norm(name2_embedding))
                name1_embedding_trans = self.mlp_p(name1_embedding.view(-1,1)).view(name1_embedding.size())
                name2_embedding_trans = self.mlp_p(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                inferences += self.FC[name1 + ":" + name2](OPS[self.arch['binary']](name1_embedding_trans, name2_embedding_trans))
        return inferences, regs

class Network_Search(Virtue):

    def __init__(self, embedding_dim, reg):
        super(Network_Search, self).__init__(embedding_dim, reg)
        self.FC = nn.ModuleDict({})
        for name1 in self.columns:
            for name2 in self.columns:
                temp = nn.ModuleList()
                for primitive in PRIMITIVES_BINARY:
                    if primitive == 'concat':
                        temp.append(nn.Linear(2*embedding_dim, 2, bias=False))
                    else:
                        temp.append(nn.Linear(embedding_dim, 2, bias=False))
                self.FC[name1 + ":" + name2] = temp
        self._initialize_alphas()

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self._arch_parameters = {}
        self._arch_parameters['mlp'] = {}
        self._arch_parameters['mlp']['p'] = self.mlp_p
        self._arch_parameters['mlp']['q'] = self.mlp_q
        self._arch_parameters['binary'] = Variable(torch.ones(len(PRIMITIVES_BINARY),
                                                              dtype=torch.float, device='cpu') / 2, requires_grad=True)
        #self._arch_parameters['binary'] = Variable(torch.Tensor([1.0,1.0,1.0,1.0,1.0]), requires_grad=True)
        self._arch_parameters['binary'].data.add_(
            torch.randn_like(self._arch_parameters['binary'])*1e-3)

    def arch_parameters(self):
        return list(self._arch_parameters['mlp']['p'].parameters()) + \
               list(self._arch_parameters['mlp']['q'].parameters()) + [self._arch_parameters['binary']]

    def new(self):
        model_new = Network_Search(self.num_users, self.num_items, self.embedding_dim, self.reg)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new

    def clip(self):
        m = nn.Hardtanh(0, 1)
        self._arch_parameters['binary'].data = m(self._arch_parameters['binary'])

    def binarize(self):
        self._cache = self._arch_parameters['binary'].clone()
        max_index = self._arch_parameters['binary'].argmax().item()
        for i in range(self._arch_parameters['binary'].size(0)):
            if i == max_index:
                self._arch_parameters['binary'].data[i] = 1.0
            else:
                self._arch_parameters['binary'].data[i] = 0.0

    def recover(self):
        self._arch_parameters['binary'].data = self._cache
        del self._cache

    def forward(self, features):
        # for i in range(len(PRIMITIVES_BINARY)):
        #     constrain(next(self._FC[i].parameters()))
        inferences = 0
        regs = 0
        for name1 in self.columns:
            for name2 in self.columns:
                name1_embedding = self.embedding_all[name1](features[name1])
                name2_embedding = self.embedding_all[name2](features[name2])
                regs += self.reg * (torch.norm(name1_embedding) + torch.norm(name2_embedding))
                name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                name2_embedding_trans = self.mlp_p(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, self._arch_parameters['binary'], self.FC[name1 + ":" + name2])
        return inferences, regs

    def genotype(self):
        genotype = PRIMITIVES_BINARY[self._arch_parameters['binary'].argmax().cpu().numpy()]
        genotype_p = F.softmax(self._arch_parameters['binary'], dim=-1)
        return genotype, genotype_p.cpu().detach()

    def step(self, features, features_valid, lr, arch_optimizer, unrolled):
        self.zero_grad()
        arch_optimizer.zero_grad()
        # binarize before forward propagation
        self.binarize()
        loss = self._backward_step(features_valid)
        # restore weight before updating
        self.recover()
        arch_optimizer.step()
        return loss

    def _backward_step(self, features_valid):
        inferences, regs = self(features_valid)
        loss = self.compute_loss(inferences, features_valid["label"], regs)
        loss.backward()
        return loss

class DSNAS(Virtue):

    def __init__(self, embedding_dim, reg, args):
        super(DSNAS, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self.FC = nn.ModuleDict({})
        for name1 in self.columns:
            for name2 in self.columns:
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
            self.log_alpha = torch.nn.Parameter(torch.zeros((num_op*num_op, len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        else:
            self.log_alpha = torch.nn.Parameter(torch.zeros((1, len(PRIMITIVES_BINARY))).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        self._arch_parameters = [self.log_alpha]
        self.weights = Variable(torch.zeros_like(self.log_alpha))
        if self.args.early_fix_arch:
            self.fix_arch_index = {}

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
        for name1 in self.columns:
            for name2 in self.columns:
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
                name1_embedding_trans = name1_embedding#self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                name2_embedding_trans = name2_embedding#self.mlp_p(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        #ipdb.set_trace()
        pos_weights = torch.zeros_like(features["label"])
        max_index = torch.argmax(inferences, dim=1)
        simulate_index = []
        for i in range(len(max_index)):
            if np.random.random() < self.args.epsion:
                simulate_index.append(np.random.randint(0, 2))
            else:
                simulate_index.append(max_index[i])
        a_ind = np.array([(i, val) for i, val in enumerate(simulate_index)])
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

    def step(self, optimizer, arch_optimizer):
        self.train()
        losses = []
        train_bandit = utils.get_data_queue_bandit(self.args, self.contexts, self.pos_weights)
        cnt = 0
        time_data = 0
        time_forward = 0
        time_update = 0
        end = -1
        for step, features in enumerate(train_bandit):
            if end!=-1:
                time_data += time.time() - end
            begin = time.time()
            optimizer.zero_grad()
            arch_optimizer.zero_grad()
            output, error_loss, loss_alpha = self.forward(features)
            time_forward += time.time() - begin
            losses.append(error_loss.cpu().detach().item())
            begin2 = time.time()
            optimizer.step()
            arch_optimizer.step()
            time_update += time.time() - begin2
            cnt += 1
            end = time.time()
        print("time_data: ", time_data)
        print("time_forward: ", time_forward)
        print("time_update: ", time_update)
        print("cnt: ", cnt)
        return np.mean(losses)

    def revised_arch_index(self):
        if self.args.early_fix_arch:
            sort_log_alpha = torch.topk(F.softmax(self.log_alpha.data, dim=-1), 2)
            argmax_index = (sort_log_alpha[0][:, 0] - sort_log_alpha[0][:, 1] >= 0.01)
            for id in range(argmax_index.size(0)):
                if argmax_index[id] == 1 and id not in self.fix_arch_index.keys():
                    self.fix_arch_index[id] = [sort_log_alpha[1][id, 0].item(),
                                                self.log_alpha.detach().clone()[id, :]]
            for key, value_lst in self.fix_arch_index.items():
                self.log_alpha.data[key, :] = value_lst[1]

    def forward(self, features):
        regs = 0
        self.weights = self._get_weights(self.log_alpha)

        self.revised_arch_index()
        if self.args.early_fix_arch:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.weights[key, :].zero_()
                    self.weights[key, value_lst[0]] = 1

        cate_prob = F.softmax(self.log_alpha, dim=-1)
        self.cate_prob = cate_prob.clone().detach()
        loss_alpha = torch.log(
            (self.weights * F.softmax(self.log_alpha, dim=-1)).sum(-1)).sum()
        self.weights.requires_grad_()

        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0

        from sklearn.externals.joblib import Parallel, delayed
        names_all = []
        for name1 in self.columns:
            for name2 in self.columns:
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
                names_all.append([name1_embedding, name2_embedding, cur_weights.view(-1,), self.FC[name1 + ":" + name2]])
        res = Parallel(n_jobs=8, backend="threading")(delayed(MixedBinary)(para1, para2, para3, para4) for para1,para2,para3,para4 in names_all)
        inferences = sum(res)
        # for name1 in self.columns:
        #     for name2 in self.columns:
        #         if self.args.multi_operation:
        #             cur_weights = self.weights[cur_index]
        #             max_index = cur_weights.argmax().item()
        #             cur_index += 1
        #         if self.args.ofm:
        #             name1_embedding = self.embedding_all[name1][max_index](features[name1])
        #             name2_embedding = self.embedding_all[name2][max_index](features[name2])
        #         else:
        #             name1_embedding = self.embedding_all[name1](features[name1])
        #             name2_embedding = self.embedding_all[name2](features[name2])
        #         regs += self.reg * (torch.norm(name1_embedding) + torch.norm(name2_embedding))
        #         name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
        #         name2_embedding_trans = self.mlp_p(name2_embedding.view(-1, 1)).view(name2_embedding.size())
        #         inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        self.weights.grad = torch.zeros_like(self.weights)
        (weighted_loss + loss_alpha).backward()
        self.block_reward = self.weights.grad.data.sum(-1)
        self.log_alpha.grad.data.mul_(self.block_reward.view(-1, 1))

        return inferences, weighted_loss, loss_alpha

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

class Uniform:
    def __init__(self, embedding_dim, reg, args):
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])

    def recommend(self, features):
        pos_weights = torch.zeros_like(features["label"])
        max_index = np.random.randint(0, 2, features["label"].shape[0])
        a_ind = np.array([(i, val) for i, val in enumerate(max_index)])
        pos_weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        reward = torch.sum(torch.mul(features["label"], pos_weights)).cpu().detach().item()
        return reward

    def step(self, optimizer, arch_optimizer):
        return 0

    def genotype(self):
        return "uniform", 0

class Egreedy:
    def __init__(self, embedding_dim, reg, args):
        self.log_alpha = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        self.epsion = 0.2
        self.action_rewards = {0:[0,1.0], 1:[0,1.0]}#total reward, action_num
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

    def step(self, optimizer, arch_optimizer):
        return 0

    def genotype(self):
        return "uniform", 0

class FM(Virtue):

    def __init__(self, embedding_dim, reg, args):
        super(FM, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self.FC = nn.ModuleDict({})
        for name1 in self.columns:
            for name2 in self.columns:
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
        self.log_alpha = torch.Tensor([1, 1, 1, 1, 1.0])
        self.weights = Variable(torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0]))

    def recommend(self, features):
        self.eval()
        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0
        for name1 in self.columns:
            for name2 in self.columns:
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
                name1_embedding_trans = name1_embedding#self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                name2_embedding_trans = name2_embedding#self.mlp_p(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
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

    def step(self, optimizer, arch_optimizer):
        self.train()
        losses = []
        train_bandit = utils.get_data_queue_bandit(self.args, self.contexts, self.pos_weights)
        cnt = 0
        for step, features in enumerate(train_bandit):
            optimizer.zero_grad()
            output, error_loss, loss_alpha = self.forward(features)
            losses.append(error_loss.cpu().detach().item())
            optimizer.step()
            cnt += 1
        print("cnt: ", cnt)
        return np.mean(losses)

    def forward(self, features):
        regs = 0
        inferences = 0
        max_index = self.weights.argmax().item()
        cur_weights = self.weights
        cur_index = 0
        for name1 in self.columns:
            for name2 in self.columns:
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
                regs += self.reg * (torch.norm(name1_embedding) + torch.norm(name2_embedding))
                name1_embedding_trans = self.mlp_p(name1_embedding.view(-1, 1)).view(name1_embedding.size())
                name2_embedding_trans = self.mlp_p(name2_embedding.view(-1, 1)).view(name2_embedding.size())
                inferences += MixedBinary(name1_embedding_trans, name2_embedding_trans, cur_weights.view(-1,), self.FC[name1 + ":" + name2])
        loss = (inferences - features["label"])**2
        weighted_loss = torch.mean(torch.sum(torch.mul(features["pos_weights"], loss), dim=1))
        weighted_loss.backward()
        return inferences, weighted_loss, 0

    def genotype(self):
        return "FM", 0

class Plus(FM):
    def __init__(self, embedding_dim, reg, args):
        super(Plus, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self._initialize_alphas()

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.weights = Variable(torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

    def genotype(self):
        return "Plus", 0


class Max(FM):
    def __init__(self, embedding_dim, reg, args):
        super(Max, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self._initialize_alphas()

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.weights = Variable(torch.Tensor([0.0, 0.0, 1.0, 0.0, 0.0]))

    def genotype(self):
        return "Max", 0

class Min(FM):
    def __init__(self, embedding_dim, reg, args):
        super(Min, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self._initialize_alphas()

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.weights = Variable(torch.Tensor([0.0, 0.0, 0.0, 1.0, 0.0]))

    def genotype(self):
        return "Min", 0

class Concat(FM):
    def __init__(self, embedding_dim, reg, args):
        super(FM, self).__init__(embedding_dim, reg, ofm=args.ofm, embedding_num=args.embedding_num)
        self._initialize_alphas()

    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.mlp_q = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1))
        self.weights = Variable(torch.Tensor([0.0, 0.0, 0.0, 0.0, 1.0]))

    def genotype(self):
        return "Concat", 0