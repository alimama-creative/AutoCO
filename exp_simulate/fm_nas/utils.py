from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    """ Implements FTRL online learning algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter (default: 1.0)
        beta (float, optional): beta parameter (default: 1.0)
        l1 (float, optional): L1 regularization parameter (default: 1.0)
        l2 (float, optional): L2 regularization parameter (default: 1.0)
    .. _Ad Click Prediction: a View from the Trenches: 
        https://www.eecs.tufts.edu/%7Edsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, params, alpha=1.0, beta=1.0, l1=1.0, l2=1.0):
        if not 0.0 < alpha:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not 0.0 < beta:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= l1:
            raise ValueError("Invalid l1 parameter: {}".format(l1))
        if not 0.0 <= l2:
            raise ValueError("Invalid l2 parameter: {}".format(l2))

        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super(FTRL, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)

                z, n = state["z"], state["n"]

                theta = (n + grad ** 2).sqrt() / group["alpha"] - n.sqrt()
                z.add_(grad - theta * p.data)
                n.add_(grad ** 2)

                p.data = (
                    -1
                    / (group["l2"] + (group["beta"] + n.sqrt()) / group["alpha"])
                    * (z - group["l1"] * z.sign())
                )
                p.data[z.abs() < group["l1"]] = 0

        return loss

class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def cal_group_auc(labels, preds, user_id_list):
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc

