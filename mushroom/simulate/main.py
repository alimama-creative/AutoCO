import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
import time
import utils
from train import train
from models import DSNAS
from vartional_model import DSNAS_v, NASP_v, MAX_v, PLUS_v, CONCAT_v, MIN_v, MULTIPLY_v, NASP
from baseline import FM, FM_v, FM_v2, Random, Egreedy, Thompson, LinUCB, LinEGreedy, LinThompson

os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser(description="Search.")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--lr', type=float, default=5e-2, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--batch_size', type=int, default=512, help='choose batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--search_epochs', type=int, default=100, help='num of searching epochs')
parser.add_argument('--train_epochs', type=int, default=10000, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='sif', help='choose how to search')
parser.add_argument('--embedding_dim', type=int, default=8, help='dimension of embedding')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--gen_max_child', action='store_true', default=False, help='generate child network by argmax(alpha)')
parser.add_argument('--gen_max_child_flag', action='store_true', default=False, help='flag of generating child network by argmax(alpha)')
parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
parser.add_argument('--early_fix_arch', action='store_true', default=False, help='bn affine flag')
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
parser.add_argument('--ofm', action='store_true', default=False, help="different operation with different embedding")
parser.add_argument('--embedding_num', type=int, default=12, help="the size of embedding dictionary")
parser.add_argument('--multi_operation', action='store_true', default=False, help="use multi operation or not")
parser.add_argument('--epsion', type=float, default=0.0, help="epsion of egreedy")
parser.add_argument('--search_epoch', type=int, default=10, help="the epoch num for searching arch")
parser.add_argument('--trans', action='store_true', default=False, help="trans the embedding or not!")
parser.add_argument('--first_order', action='store_true', default=False, help="use first order or not!")
args = parser.parse_args()
print("args ofm:", args.ofm)
print("embedding_num:", args.embedding_num)
save_name = 'experiments/{}/search-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.dataset, time.strftime("%Y%m%d-%H%M%S"),
	args.mode, args.save, args.embedding_dim, args.opt, args.lr, args.arch_lr, args.seed)
if args.unrolled:
    save_name += '-unrolled'
save_name += '-' + str(np.random.randint(10000))
utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    # torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    data_start = time.time()
    if args.mode == "LinUCB" or args.mode == "LinEGreedy" or args.mode == "LinThompson":
        train_queue = utils.get_data_queue(args, True)
    else:
        train_queue = utils.get_data_queue(args, False)
    logging.info('prepare data finish! [%f]' % (time.time() - data_start))

    if args.mode == "DSNAS":
        model = DSNAS(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'], args.lr)
        arch_optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters() if name == 'log_alpha'],
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=0.0
        )
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info('genotype: %s' % g)
            logging.info('genotype_p: %s' % gp)
    elif args.mode == "DSNAS_v":
        model = DSNAS_v(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters() if name == 'log_alpha'],
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=0.0
        )
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info('genotype: %s' % g)
            logging.info('genotype_p: %s' % gp)
    elif args.mode == "NASP_v":
        model = NASP_v(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters() if name == 'log_alpha'],
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=0.0
        )
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info('genotype: %s' % g)
            logging.info('genotype_p: %s' % gp)
    elif args.mode == "NASP":
        model = NASP(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters() if name == 'log_alpha'],
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=0.0
        )
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info('genotype: %s' % g)
            logging.info('genotype_p: %s' % gp)
    elif args.mode == "Random":
        model = Random(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = None
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "Egreedy":
        model = Egreedy(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = None
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "Thompson":
        model = Thompson(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = None
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "LinUCB":
        model = LinUCB(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = None
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "LinThompson":
        model = LinThompson(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = None
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "LinEGreedy":
        model = LinEGreedy(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = None
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "FM":
        model = FM(args.embedding_dim, args.weight_decay, args)
        g, gp = model.genotype()
        logging.info('genotype: %s' % g)
        logging.info('genotype_p: %s' % gp)

        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'], args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "FM_v":
        model = FM_v(args.embedding_dim, args.weight_decay, args)
        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "MULTIPLY_v":
        model = MULTIPLY_v(args.embedding_dim, args.weight_decay, args)
        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "MAX_v":
        model = MAX_v(args.embedding_dim, args.weight_decay, args)
        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "MIN_v":
        model = MIN_v(args.embedding_dim, args.weight_decay, args)
        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "PLUS_v":
        model = PLUS_v(args.embedding_dim, args.weight_decay, args)
        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    elif args.mode == "CONCAT_v":
        model = CONCAT_v(args.embedding_dim, args.weight_decay, args)
        optimizer = torch.optim.Adagrad([param for name, param in model.named_parameters() if name != 'log_alpha'],
                                        args.lr)
        arch_optimizer = None
        for search_epoch in range(1):
            g, gp, loss, rewards = train(train_queue, model, optimizer, arch_optimizer, logging)
            logging.info("total reward: %s" % rewards)
    else:
        raise ValueError("bad choice!")

#python main.py --mode="sif" --dataset="mushroom"
#python main.py --mode="DSNAS" --dataset="mushroom" --gen_max_child --early_fix_arch --arch_lr=0.001
if __name__ == '__main__':
    main()
