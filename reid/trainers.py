from __future__ import print_function, absolute_import

from torch.autograd import Variable
import torch
import time

from .loss import TripletLoss, CETLossV2, CETCTLoss
from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter

class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):

        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]

        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            targets = Variable(pids.to(mps_device))
        else:
            targets = Variable(pids.cuda())
        
        return inputs, targets

    def _forward(self, inputs, targets):
        features, _, logits = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(logits, targets)
            prec, = accuracy(logits.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(features, targets)
        elif isinstance(self.criterion, CETLossV2):
            loss = self.criterion(features, logits, targets)
            prec, = accuracy(logits.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, CETCTLoss):
            loss = self.criterion(features, logits, targets)
            prec, = accuracy(logits.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
