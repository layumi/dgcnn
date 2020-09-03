#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN
from OGNet import Model_dense
from model_efficient import ModelE_dense
from pointnet2_model import PointNet2SSG, PointNet2MSG
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from ptflops import get_model_complexity_info

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'ssg': 
        model = PointNet2SSG( output_classes=40, dropout_prob=args.dropout)
        model.to(device)
    elif args.model == 'msg':
        model = PointNet2MSG( output_classes=40, dropout_prob=args.dropout)
        model.to(device)
    elif args.model == 'ognet':
        # [64,128,256,512]
        model = Model_dense(20, args.feature_dims, [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=args.dropout, id_skip = args.id_skip, drop_connect_rate=args.drop_connect_rate,cluster='xyzrgb', pre_act = args.pre_act, norm = args.norm_layer)
        if args.efficient:
             model = ModelE_dense(20, args.feature_dims, [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=args.dropout, id_skip = args.id_skip, drop_connect_rate=args.drop_connect_rate,cluster='xyzrgb', pre_act = args.pre_act, norm = args.norm_layer)
        model.to(device)
    elif args.model == 'ognet-small':
        # [48,96,192,384] 
        model = Model_dense(20, args.feature_dims, [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=args.dropout, id_skip = args.id_skip,drop_connect_rate=args.drop_connect_rate, cluster='xyzrgb', pre_act = args.pre_act , norm = args.norm_layer)
        model.to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    best_avg_per_class_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            opt.zero_grad()
            if args.model == 'ognet' or args.model == 'ognet-small' or args.model=='ssg' or args.model=='msg':
                logits = model(data, data)
            else:
                data = data.permute(0, 2, 1)
                logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            if args.model == 'ognet' or args.model == 'ognet-small' or args.model=='ssg' or args.model=='msg':
                logits = model(data, data)
            else:
                data = data.permute(0, 2, 1)
                logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc + avg_per_class_acc >= best_test_acc + best_avg_per_class_acc:
            best_test_acc = test_acc
            best_avg_per_class_acc = avg_per_class_acc
            print('This is the current best.')
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'ssg':
        model = PointNet2SSG( output_classes=40, dropout_prob=0)
        model.to(device)
    elif args.model == 'msg':
        model = PointNet2MSG( output_classes=40, dropout_prob=0)
        model.to(device)
    elif args.model == 'ognet':
        # [64,128,256,512]
        model = Model_dense(20, args.feature_dims, [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=args.dropout, id_skip = args.id_skip, drop_connect_rate=args.drop_connect_rate,cluster='xyzrgb', pre_act = args.pre_act, norm = args.norm_layer)
        if args.efficient:
             model = ModelE_dense(20, args.feature_dims, [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=args.dropout, id_skip = args.id_skip, drop_connect_rate=args.drop_connect_rate,cluster='xyzrgb', pre_act = args.pre_act, norm = args.norm_layer)
        model.to(device)
    elif args.model == 'ognet-small':
        # [48,96,192,384]
        model = Model_dense(20, args.feature_dims, [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=args.dropout, id_skip = args.id_skip,drop_connect_rate=args.drop_connect_rate, cluster='xyzrgb', pre_act = args.pre_act , norm = args.norm_layer)
        model.to(device)
    else:
        raise Exception("Not implemented")

    try:
        model.load_state_dict(torch.load(args.model_path))
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    model = model.module

    batch0,label0 = next(iter(test_loader))
    batch0 = batch0[0].unsqueeze(0)
    print(batch0.shape)
    print(model)
    
    macs, params = get_model_complexity_info(model, batch0, ( (1024, 3) ), as_strings=True, print_per_layer_stat=False, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        batch_size = data.size()[0]
        if args.model == 'ognet' or args.model == 'ognet-small' or args.model=='ssg' or args.model=='msg':
            logits = model(data, data) 
            #logits = model(1.1*data, 1.1*data)
        else:
            data = data.permute(0, 2, 1)
            logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'ognet', 'ognet-small', 'ssg', 'msg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--drop_connect_rate', type=float, default=0.5,
                        help='drop connect rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--feature_dims',default='64,128,256,512', type=str,
                        help='64, 128, 256, 512 or 64, 128, 256, 512, 1024')
    parser.add_argument('--id_skip', action='store_true')
    parser.add_argument('--efficient', action='store_true')
    parser.add_argument('--pre_act', action='store_true')
    parser.add_argument('--norm_layer', type=str, default='bn')
    args = parser.parse_args()

    _init_()

    if len(args.feature_dims)>0:
        str_features = args.feature_dims.split(',')
        features = []
        for feature in str_features:
            feature = int(feature)
            features.append(feature)
        args.feature_dims = features

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
