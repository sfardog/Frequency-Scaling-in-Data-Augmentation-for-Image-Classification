import os
import random
import argparse

import torch
import torchvision.transforms as transforms

import numpy as np

def argparsing() :
    parser = argparse.ArgumentParser(
        description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--augment', type=str, default='original', help='augmenting method')
    parser.add_argument('--data_path', type=str, default='CIFAR/dataset')
    parser.add_argument('--data_type', type=str, default='cifar10')
    parser.add_argument('--parallel', action='store_true', default=True)
    parser.add_argument('--image_size', type=int, default=32, help='image size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')

    # Setting hyperparameters
    parser.add_argument('--model_name', type=str, default='densenet121')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # Print parameter
    parser.add_argument('--step', type=int, default=50)

    args = parser.parse_args()

    return args

def get_deivce() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def fix_seed(device) :
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)

    if device == 'cuda' :
        torch.cuda.manual_seed_all(4321)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("your seed is fixed to '4321'")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def transform_generator(args) :
    test_transform_list = [
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ]

    return transforms.Compose(test_transform_list)

def get_save_path(args, model_dir = './CIFAR/test_result_save') :
    model_dir = os.path.join(model_dir, args.data_type, 'pretrained=False')
    model_dirs = os.path.join(model_dir, str(args.image_size), args.optimizer, str(args.batch_size), args.augment)
    if not os.path.exists(os.path.join(model_dirs)): os.makedirs(os.path.join(model_dirs))
    save_model_path = '{}_{}'.format(args.lr, str(args.epochs).zfill(3))

    return model_dirs, save_model_path

def get_load_path(args, eecp, model_dir='./model_save') :
    if eecp:
        model_name = '{}(k_clusters:{}, length:{}, angle:{}, lamb:{}, preserve_range:{}, weight_factor : {}, combination:{})'.format(
            args.model_name, args.k_clusters, args.length, args.angle, args.lamb, args.preserve_range, args.weight_factor,
            args.combination)
    else:
        model_name = args.model_name

    model_dir = os.path.join(model_dir, args.data_type, 'pretrained=False')
    model_dirs = os.path.join(model_dir, str(args.image_size), args.optimizer, str(args.batch_size), args.augment, model_name)
    load_model_path = '{}_{}.pth'.format(args.lr, str(args.epochs).zfill(3))
    load_model_path = os.path.join(model_dirs, load_model_path)

    return load_model_path