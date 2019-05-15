import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.add_argument("-n_classes", type=int, help="number of classes", default=40)
parser.add_argument("-n_epochs", type=int, help="number of epochs", default=30)
parser.add_argument("-classnames", type=str, help="class names", default=None)
parser.add_argument("-no_cuda", dest='no_cuda', action='store_true')
parser.add_argument("-init_from", type=str, help="init with pretrained model", default=None)
parser.add_argument("-max_batches", type=int, help="number of epochs", default=None)
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

def train_stage1(args):
    if args.init_from is None:
        log_dir = args.name+'_stage_1'
    else:
        log_dir = None
    pretraining = not args.no_pretraining
    classnames = args.classnames.split(',')
    n_models_train = args.num_models*args.num_views

    if log_dir is not None:
        create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=args.n_classes, pretraining=pretraining,
                 cnn_name=args.cnn_name, no_cuda=args.no_cuda)

    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

    print("train_path: {}".format(args.train_path))
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False,
                                     num_models=n_models_train, num_views=args.num_views,
                                     classnames=classnames)
    print("train_dataset: {}".format(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                               num_workers=0)

    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False,
                                   test_mode=True, classnames=classnames)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1, n_classes=args.n_classes, no_cuda=args.no_cuda)
    trainer.train(args.n_epochs, args.max_batches)

    return cnet

def train_stage2(cnet, args):
    if args.init_from is None:
        log_dir = args.name+'_stage_2'
    else:
        log_dir = None
    classnames = args.classnames.split(',')
    n_models_train = args.num_models*args.num_views

    if log_dir is not None:
        create_folder(log_dir)
    if args.init_from is None:
        cnet_2 = MVCNN(args.name, cnet, nclasses=args.n_classes, cnn_name=args.cnn_name,
                       num_views=args.num_views, no_cuda=args.no_cuda)
        del cnet
    else:
        cnet_2 = torch.load(args.init_from)

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                           betas=(0.9, 0.999))
    
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False,
                                        num_models=n_models_train, num_views=args.num_views, classnames=classnames)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,
                                               shuffle=False, num_workers=0) # shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False,
                                      num_views=args.num_views, classnames=classnames)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,
                                             shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                              'mvcnn', log_dir, num_views=args.num_views, n_classes=args.n_classes,
                              no_cuda=args.no_cuda)
    trainer.train(args.n_epochs, args.max_batches)

if __name__ == '__main__':
    args = parser.parse_args()

    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    if args.init_from is None:
        cnet = train_stage1(args)
    else:
        cnet = None
    train_stage2(cnet, args)
