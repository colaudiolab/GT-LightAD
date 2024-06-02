import argparse
import torch
from exp.exp_main import Exp_Main
import numpy as np

parser = argparse.ArgumentParser(description='standard model')
parser.add_argument('--model', type=str,  default='FModel')

# data loader
parser.add_argument('--dataset', type=str, default='MSL', help='data file')
parser.add_argument('--data_path', type=str, default='data/MSL', help='root path of the data file')
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--anormly_ratio', type=float, default=2)

# forecasting task
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
parser.add_argument('--pre_len', type=int, default=100, help='prediction sequence length')
parser.add_argument('--feature_dimension', type=int, default=55, help='num of feature')

# model
parser.add_argument('--MA_factor', type=int, default=20, help='optimizer learning rate')
parser.add_argument('--layer', type=int, default=5, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='self attention dropout')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.mode == 'train':
    setting = '{}_{}_{}_{}_{}'.format(
    args.dataset,
    args.layer,
    args.MA_factor,
    args.batch_size,
    args.learning_rate)

    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

if args.mode == 'test':
    for itr in range(args.itr):
        setting = '{}_{}_{}_{}_{}'.format(
        args.dataset,
        args.layer,
        args.MA_factor,
        args.batch_size,
        args.learning_rate)

        exp = Exp(args)
        print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()

