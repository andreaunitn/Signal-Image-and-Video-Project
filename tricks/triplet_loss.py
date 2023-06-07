from __future__ import print_function, absolute_import
import os.path as osp
import argparse

from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch import nn
import numpy as np
import torch
import sys

from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from reid.dist_metric import DistanceMetric
from reid.loss import CETLossV2, CETCTLoss
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.trainers import Trainer
from reid import datasets
from reid import models

import warnings
warnings.filterwarnings("ignore")


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances, workers, combine_trainval, tricks):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval else dataset.num_train_ids)

    # -----------------------------
    # Trick 2: Random Erasing Augmentation
    if tricks < 2:
        train_transformer = T.Compose([
            T.RandomSizedRectCrop(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])
    else:
        train_transformer = T.Compose([
            T.RandomSizedRectCrop(height, width), 
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            T.RandomErasingAugmentation(height, width),
        ])
    # -----------------------------

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer), 
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)), root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    argv = sys.argv

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'), "".join(argv))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, 'num_instances should divide batch_size'
    
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    
    dataset, num_classes, train_loader, val_loader, test_loader = get_data(args.dataset, args.split, args.data_dir, args.height, args.width, args.batch_size, args.num_instances, args.workers, args.combine_trainval, args.t)

    # -----------------------------
    # Trick 4: Last Stride
    if args.t < 4:
        last_stride_value = 2
    else:
        last_stride_value = 1
    # -----------------------------
    
    # -----------------------------
    # Trick 5: BNNeck
    if args.t < 5:
        norm = False
    else:
        norm = True
        args.dist_metric = "cosine"
    # -----------------------------

    # Create model
    model = models.create(args.arch, dropout=args.dropout, num_classes=num_classes, last_stride=last_stride_value, norm=norm)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}".format(start_epoch, best_top1))

    # Enabling GPU acceleration on Mac devices
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        model = nn.DataParallel(model).to(mps_device)
    else:
        model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # -----------------------------
    # Re-ranking
    if args.re_ranking:
        re_ranking = True
    else:
        re_ranking = False
    # -----------------------------

    # Evaluator
    evaluator = Evaluator(model)

    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val, metric, norm=norm, re_ranking=re_ranking)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, norm=norm, re_ranking=re_ranking)
        return

    # -----------------------------
    # Trick 3: Label Smoothing
    if args.t < 3:
        # Criterion
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            criterion = CETLossV2(num_classes, margin=args.margin).to(mps_device)
        else:
            criterion = CETLossV2(num_classes, margin=args.margin).cuda()
    elif 3 <= args.t <= 5:
        # Criterion
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            criterion = CETLossV2(num_classes, margin=args.margin, e=0.1).to(mps_device)
        else:
            criterion = CETLossV2(num_classes, margin=args.margin, e=0.1).cuda()
    else:
        # Criterion
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            criterion = CETCTLoss(num_classes, feat_dim=2048, margin=args.margin, e=0.1).to(mps_device)
        else:
            criterion = CETCTLoss(num_classes, feat_dim=2048, margin=args.margin, e=0.1).cuda()
    # -----------------------------

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # -----------------------------
    # Trick 1: Warmup Learning Rate
    def adjust_lr(epoch):

        if args.t < 1:
            if epoch <= 39:
                lr = args.lr
            elif 40 <= epoch <= 69:
                lr = args.lr * 0.1
            else:
                lr = args.lr * 0.1 * 0.1
        else:
            if epoch <= 10:
                lr = args.lr * (epoch * 0.1)
            elif 11 <= epoch <= 40:
                lr = args.lr
            elif 41 <= epoch <= 70:
                lr = args.lr * 0.1
            else:
                lr = args.lr * 0.1 * 0.1

        
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
    # -----------------------------

    # Start training
    for epoch in range(start_epoch + 1, args.epochs + 1):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        
        if epoch < args.start_save:
            continue
        
        top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val, norm=norm, re_ranking=re_ranking)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, norm=norm, re_ranking=re_ranking)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Person re-identification training and evaluation parameters")
    
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, help="input width, default: 128 for resnet*")
    parser.add_argument('--combine-trainval', action='store_true', help="train and validation sets together for training, test set alone for evaluation")
    parser.add_argument('--num-instances', type=int, default=4, help="each minibatch consist of (batch_size // num_instances) identities, and each identity has num_instances instances, default: 4")
    
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    # loss
    parser.add_argument('--margin', type=float, default=0.3, help="margin of the triplet loss, default: 0.3")
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate of all parameters")
    
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--start_save', type=int, default=0, help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))

    # re-ranking
    parser.add_argument("--re_ranking", type=bool, default=False)

    # trick number
    parser.add_argument('-t', type=int, default=0)
   
    main(parser.parse_args())