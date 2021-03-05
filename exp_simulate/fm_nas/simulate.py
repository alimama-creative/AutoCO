import io
import os
import time
import pickle
import random
import logging
import datetime
import argparse
import coloredlogs
import numpy as np
import pandas as pd

from torch.utils.data import WeightedRandomSampler
from policy import FmEGreedy, Random, Greedy, FmGreedy, LinUCB, FmThompson, TS


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def click(x, k, item_ctr, r_t):
    "return the reward for impression"
    if r_t == "click":
        t = random.uniform(0,1)
        if item_ctr[x][k] > t:
            return 1
        return 0 
    else:
        return item_ctr[x][k]


def process_batch(features_list, memory, policy, item_ctr):
    "recommend items under current policy"
    # data: the impression items in current batch
    data = []
    for x in memory:
        t = [x]
        t.extend(features_list[x])
        data.append(t)
    # policy recommend creative
    if policy.name == 'Greedy':
        res = policy.recommend_batch(data, item_ctr)
    else:
        res = policy.recommend_batch(data)
    return res


def evaluate(policy, params):
    "process of simulation"

    # initial reward recorder var
    record_arr = [0.5,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,36,40,45,50,60,70,80,90,
                  100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,
                  260,280,300,330,360,400,500]
    score, impressions = 0.0, 1.0
    ctr_reward, auc_reward = [], []

    # initial data recorder var
    memory, record, r_list, eg_ind = [], [], [], []
    cnt = 0
    initial = 0

    # initial background information
    f = open(params["feature_list"], 'rb')
    f_list = pickle.load(f)
    f = open(params["pro_list"], 'rb')
    pro_list = pickle.load(f)
    f = open(params['creative_list'], 'rb')
    c_list = pickle.load(f)
    f = open(params['item_candidates'], 'rb')
    item_candi = pickle.load(f)
    f = open(params['item_ctr'], 'rb')
    item_ctr = pickle.load(f)
    f_len = len(f_list[0])
    leng = f_len+ len(c_list[0])
    item_cnt = params['batch']
    df = pd.read_pickle(params["random_file"])
    warm_start = list(df.to_numpy())
    record = warm_start[200000:202000]

    # the main process of simulation
    while impressions <= params["iter"]:
        cnt += 1
        item_cnt += 1
        # decide which item to display
        if item_cnt >= params['batch']:
            item_list = list(WeightedRandomSampler(pro_list, params['batch']))
            item_cnt = 0
        x = item_list[item_cnt]

        # if cnt < params["batch"]:
        #     line = f_list[x].copy()
        #     k = np.random.randint(0, len(item_candi[x]))
        #     line.extend(c_list[item_candi[x][k]])
        #     line.append(click(x, k, item_ctr, params["simulate_type"]))
        #     record.append(line)
        #     eg_ind.append(x)
        #     continue

        # update policy with batch data
        if len(record) >= params['batch']-3 or initial == 0:
            initial = 1
            auc_reward.append(policy.update(record, eg_ind))
            record = []
            eg_ind = []

        # collect reward in current batch
        memory.append(x)
        if len(memory) % params['s_batch']== 0 and len(memory)>0 :      
            r_list = process_batch(f_list, memory, policy, item_ctr)
            for i in range(len(r_list)):
                line = f_list[memory[i]].copy()
                t= item_candi[memory[i]][r_list[i]]
                line.extend(c_list[t])
                reward = click(memory[i], r_list[i], item_ctr, params["simulate_type"])
                line.append(reward)
                record.append(line)
                eg_ind.append(memory[i])
                score += reward
                impressions += 1
                if impressions%10000 == 0:
                    logger.debug('{} behaviour has been generated, Ctr is {}!!!'.format(impressions, score/(impressions)))
                    print(ctr_reward)
                if impressions/10000 in record_arr:
                    ctr_reward.append(score/impressions)
                # if impressions%1000000 == 0:
                #     policy.update_in_log()
            memory.clear()
    # policy.print()
    score /= impressions
    print("CTR achieved by the policy: %.5f" % score)
    return ctr_reward, auc_reward


def run(params):
    model = params['model_ee']
    if model == 'fmeg':
        policy = FmEGreedy(params)
    elif model == 'fmgreedy':
        policy = FmGreedy(params)
    elif model == 'random':
        policy = Random(params)
    elif model == 'greedy':
        policy = Greedy(params)
    elif model == 'ts':
        policy = TS(params)
    elif model == 'linucb':
        policy = LinUCB(params)
    elif model == "fmts":
        policy = FmThompson(params)
    else:
        print("No model named ", model, " !!!")
        return
    res, auc_res = evaluate(policy, params)
    return res, auc_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--info_path', type=str, default='dataset/data_nas') 
    parser.add_argument('--data_name', type=str, default="rand_0.15")
    parser.add_argument('--simulate_type', type=str, default="click")
    parser.add_argument('--iter', type=int, default=200000)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--model_ee', type=str, default='random')
    parser.add_argument('--model_nas', type=str, default="fm")
    parser.add_argument('--oper', type=str, default='multiply')
    parser.add_argument('--model_struct', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--decay', type=float, default=0.0001)
    parser.add_argument('--learning', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=5000)
    parser.add_argument('--s_batch', type=int, default=5000)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--update', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=-1)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--record', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--trick', type=int, default=0)
    parser.add_argument('--first_order', type=int, default=0)
    parser.add_argument('--calcu_dense', type=int, default=0)
    parser.add_argument('--ts_trick', type=int, default=0)
    parser.add_argument('--auc_record', type=int, default=0)

    args = parser.parse_args()
    params = {"warm_file": os.path.join(args.info_path, args.data_name,"warm_start.pkl"),
             "creative_list": os.path.join(args.info_path, "creative_list.pkl"),
             "item_candidates": os.path.join(args.info_path, "item_candidates.pkl"),
             "pro_list": os.path.join(args.info_path, "pro_list.pkl"),
             "feature_list": os.path.join(args.info_path, "feature_list.pkl"),
             "feature_size": os.path.join(args.info_path, "feature_size.pkl"),
             "item_ctr": os.path.join(args.info_path, args.data_name,"item_ctr.pkl"),
             "random_file": os.path.join(args.info_path, args.data_name,"random_log.pkl"),
             "model_ee": args.model_ee,
             'learning': args.learning,
             "batch": args.batch,
             "dim": args.dim,
             "simulate_type": args.simulate_type,
             "ts_trick": args.ts_trick,
             "arch": [0,1,0,3,0,3,0,1,4,2,0,4,3,2,0],
             "trick": args.trick,
             "model_struct": args.model_struct,
             "model_nas": args.model_nas,
             "operator": args.oper,
             "s_batch": args.s_batch,
             "epoch": args.epoch,
             "decay": args.decay,
             "device": args.device,
             "times": args.times,
             "iter": args.iter,
             "first_order": args.first_order,
             "update": args.update,
             "auc_record": args.auc_record,
             "data_size": args.data_size,
             "sample": args.sample,
             "alpha": args.alpha,
             "record": args.record,
             'optimizer': args.optimizer,
             'calcu_dense': args.calcu_dense,
             'dense': [0,6,0],
             "early_fix_arch": False,
             }

    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    params["model_path"] = os.path.join(args.info_path, args.data_name,"model",timestamp+'.pt')
    import json
    params_str = json.dumps(params)
    score, auc_score = [], []
    for i in range(params['times']):
        res, auc_res = run(params)
        score.append(res)
        filename = os.path.join(args.info_path, args.data_name,"res",params['model_ee']+"-"+timestamp+'.txt')
        # filename = 'data/res/'+params['model']+"-"+timestamp+'.txt'
        if params["record"] == 0:
            with open(filename, 'w') as f:
                f.write(params_str)
                f.write('\n')
                for i in range(len(score)):
                    s = [str(reward) for reward in score[i]]
                    f.write("time"+str(i)+": "+" ".join(s)+"\n")
        
        if params["auc_record"] == 1:
            auc_score.append(auc_res)
            filename = os.path.join(args.info_path, args.data_name,"auc_res",params['model_ee']+"-"+timestamp+'.txt')
            # filename = 'data/res/'+params['model']+"-"+timestamp+'.txt'
        
            with open(filename, 'w') as f:
                f.write(params_str)
                f.write('\n')
                for i in range(len(auc_score)):
                    s = [str(reward) for reward in auc_score[i]]
                    f.write("time"+str(i)+": "+" ".join(s)+"\n")