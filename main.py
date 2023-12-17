import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import network_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
from datasets.nyu import *
from datasets.middlebury import *
from datasets.RGBDD import *
from models import *
from model_spn.nlspnmodel import *
from math import pi, cos
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.backends.cudnn as cudnn
torch.cuda.empty_cache()

# 'NYU': '/home/qiaoxin/prj/Qiao/dataset/NYUDepthv2/'

dataset_path = {'NYU': '/home/ubuntu/new_disk/qiaoxin/data/NYU_Depthv2/',
                'Middlebury': '/home/qiaoxin/prj/Qiao/dataset/MiddleBury&Lu/Depth_Enh/01_Middlebury_Dataset/',
                'RGBDD': '/home/qiaoxin/prj/Qiao/dataset/RGBDD'}

parser = argparse.ArgumentParser(description='non-local spatial propagation network')
parser.add_argument('--name', type=str, default='MSS')
parser.add_argument('--model', type=str, default='NLSPN_mpr')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='NYU', help='NYU, Middlebury, NoisyMiddlebury, Lu')

parser.add_argument('--data_root', type=str, default=dataset_path['NYU'])
parser.add_argument('--train_batch', type=int, default=4)
parser.add_argument('--test_batch', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--epoch', default=245, type=int, help='max epoch: 198-17 | 109-11 | 380-29 | 395-30 | 245| 20')
parser.add_argument('--eval_interval',  default=10, type=int, help='eval interval')
parser.add_argument('--checkpoint',  default='scratch', type=str, help='checkpoint to use: scratch, latest, best')
parser.add_argument('--scale',  default=4, type=int, help='scale')
parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation method to generate lr depth')
parser.add_argument('--lr',  default=1e-4, type=float, help='learning rate')        # 2e-4
parser.add_argument('--lr_step',  default=100, type=float, help='learning rate decay step')
parser.add_argument('--lr_gamma',  default=1, type=float, help='learning rate decay gamma')     # 0.9
parser.add_argument('--weight_decay',  default=1e-4, type=float, help='learning rate decay gamma')
parser.add_argument('--input_size',  default=256, type=int, help='crop size for hr image')        #
parser.add_argument('--sample_q',  default=30720, type=int, help='sampled pixels per hr depth')
parser.add_argument('--noisy', action='store_true', help='add noise to train dataset')
parser.add_argument('--residual_learning', action='store_true', help='add noise to train dataset')
parser.add_argument('--test',  action='store_true', help='test mode')
parser.add_argument('--report_per_image',  action='store_true', help='report RMSE of each image')
parser.add_argument('--save',  action='store_true', help='save results')
parser.add_argument('--batched_eval',  action='store_true', help='batched evaluation to avoid OOM for large image resolution')

# non-local SPN
parser.add_argument('--prop_kernel', type=int, default=3, help='propagation kernel size')
parser.add_argument('--preserve_input', action='store_true', help='preserve input points by replacement')
parser.add_argument('--from_scratch', action='store_true', default=True, help='train from scratch')
parser.add_argument('--prop_time', type=int, default=6, help='number of propagation')
parser.add_argument('--network', type=str, default='resnet34', choices=('resnet18', 'resnet34'), help='network name')
parser.add_argument('--affinity', type=str, default='TGASS', choices=('AS', 'ASS', 'TC', 'TGASS'),
                    help='affinity type (dynamic pos-neg, dynamic pos, '
                         'static pos-neg, static pos, none')
parser.add_argument('--affinity_gamma', type=float, default=0.5,
                    help='affinity gamma initial multiplier '
                         '(gamma = affinity_gamma * number of neighbors')
parser.add_argument('--conf_prop', action='store_true', default=True, help='confidence for propagation')
parser.add_argument('--no_conf', action='store_false', dest='conf_prop', help='no confidence for propagation')
parser.add_argument('--legacy', default=False, help='legacy code support for pre-trained models')


args = parser.parse_args()
args.result = os.path.join('..', 'results')
seed_everything(args.seed)


cudnn.benchmark = True
# model
if args.model == 'pmba':
    model = Net(num_channels=1, base_filter=64,  feat=256, num_stages=3, scale_factor=args.scale)
elif args.model == 'MPR_dsr':
    model = MPRNet(args)
elif args.model == 'NLSPN_mpr':
    model = NLSPNModel(args)
elif args.model == 'JIIF':
    model = JIIF(args, 128, 128)
else:
    raise NotImplementedError(f'Model {args.model} not found')

# model.apply(network_utils.init_weights_xavier)

# loss
if args.loss == 'L1':
    criterion = nn.L1Loss()
elif args.loss == 'L2':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(f'Loss {args.loss} not found')

# dataset
if args.dataset == 'NYU':
    dataset = NYUDataset
elif args.dataset == 'Middlebury':
    dataset = MiddleburyDataset
elif args.dataset == 'RGBDD':
    dataset = RGBDD
else:
    raise NotImplementedError(f'Dataset {args.loss} not found')

if args.model in ['JIIF']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation,
                                augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale, downsample=args.interpolation,
                           augment=False, to_pixel=True, sample_q=None)  # full image
elif args.model in ['NLSPN_mpr', 'pmba', 'MPR_dsr']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation,
                                augment=True, pre_upsample=True, input_size=args.input_size, noisy=args.noisy)
        val_dataset = dataset(root=args.data_root, split='val', scale=args.scale, downsample=args.interpolation,
                                augment=True, pre_upsample=True, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale, downsample=args.interpolation,
                           augment=False, pre_upsample=True)
else:
    raise NotImplementedError(f'Dataset for model type {args.model} not found')

# dataloader
if not args.test:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False,
                                               shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False,
                                               shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, pin_memory=True, drop_last=False,
                                          shuffle=False, num_workers=args.num_workers)

# trainer
if not args.test:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)	 # base_lr = 0.1
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, cycle_mult=2.0,
                                                                 max_lr=args.lr, min_lr=5e-7, warmup_steps=5, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, objective=criterion, optimizer=optimizer, lr_scheduler=scheduler,
                      metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, eval_interval=args.eval_interval)
else:
    trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint)

# main
if not args.test:
    trainer.train(train_loader, val_loader, args.epoch)
trainer.test(test_loader)

