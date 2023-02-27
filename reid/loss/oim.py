import torch
import torch.nn.functional as F
from torch import nn, autograd

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, momentum=0.5):
        ctx.save_for_backward(inputs, targets)
        ctx.lut = lut
        ctx.momentum = momentum
        outputs = inputs.mm(lut.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        lut = ctx.lut
        momentum = ctx.momentum
        grad_inputs = grad_targets = None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(lut)

        for x, y in zip(inputs, targets):
            lut[y] = momentum * lut[y] + (1. - momentum) * x
            lut[y] /= lut[y].norm()

        return grad_inputs, grad_targets, None, None

def oim(inputs, targets, lut, momentum=0.5):
    return OIM.apply(inputs, targets, lut, momentum)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               reduction='mean' if self.size_average else 'sum')
        return loss, inputs