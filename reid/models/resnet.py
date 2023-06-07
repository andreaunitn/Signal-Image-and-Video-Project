from __future__ import absolute_import

from torchvision.models import ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights
from torchvision.models import resnet18,resnet34,resnet50,resnet101,resnet152
from torch.nn import functional as F
from torch.nn import init
from torch import nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(self, depth, weights=None, cut_at_pooling=False,
                 num_features=0, dropout=0, num_classes=0, last_stride=2, norm=False):

        if depth == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1
        elif depth == 34:
            weights = ResNet34_Weights.IMAGENET1K_V1
        elif depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V2
        elif depth == 101:
            weights = ResNet101_Weights.IMAGENET1K_V2
        elif depth == 152:
            weights = ResNet152_Weights.IMAGENET1K_V2

        super(ResNet, self).__init__()

        self.depth = depth
        self.weights = weights
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](weights=weights)

        # -----------------------------
        # Trick 4: Last Stride
        if last_stride == 1:
            self.base.layer4[0].downsample[0].stride = (1, 1)
            self.base.layer4[0].conv2.stride = (1, 1)
        # -----------------------------

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.dropout = dropout
            self.num_classes = num_classes
            self.norm = norm

            out_planes = self.base.fc.in_features
            
            # Change the num_features to CNN output channels
            self.num_features = out_planes

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes > 0:

                if self.norm:
                    self.batch_norm = nn.BatchNorm1d(self.num_features)
                    self.batch_norm.requires_grad_(False)
                    self.classifier = nn.Linear(self.num_features, self.num_classes)
                    nn.init.kaiming_normal_(self.classifier.weight, a=0, mode='fan_in')
                    nn.init.constant_(self.batch_norm.weight, 1.0)
                    nn.init.constant_(self.batch_norm.bias, 0.0)

                    #self.batch_norm.apply(weights_init_kaiming)
                    #self.classifier.apply(weights_init_classifier)
                    #init.kaiming_normal_(self.classifier.weight, mode='fan_out')
                else:
                    self.classifier = nn.Linear(self.num_features, self.num_classes)

        if self.weights is None:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        y = x.clone()

        #if self.norm:
            #x = self.batch_norm(x)
        
        z = x.clone()

        if self.dropout > 0:
            x = self.drop(x)

        if self.num_classes > 0:
            x = self.classifier(x)

        return y, z, x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
