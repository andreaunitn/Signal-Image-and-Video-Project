from torch.nn import CrossEntropyLoss
from .triplet import TripletLoss
import torch

# ------------------------------------
# Trick 3: Label Smoothing
class IDLoss(torch.nn.Module):
    def __init__(self, num_classes, epsilon=0):
        super(IDLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, labels):
        # Apply Label Smoothing
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        smooth_labels = one_hot * (1 - self.epsilon) + torch.ones_like(logits) * self.epsilon / self.num_classes

        # Compute cross entropy loss
        id_loss = torch.mean(-torch.sum(smooth_labels * torch.nn.functional.log_softmax(logits, dim = 1), dim = 1))

        return id_loss


class CETLossV2(torch.nn.Module):
    def __init__(self, num_classes, margin=0.3, e=0):
        super(CETLossV2, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.epsilon = e

        if self.epsilon == 0:
            self.cross_entropy = CrossEntropyLoss()
        else:
            self.cross_entropy = IDLoss(num_classes=self.num_classes, epsilon=self.epsilon)
            
        self.triplet = TripletLoss(margin=self.margin)

    def forward(self, features, logits, target):
        cross_entropy_loss = self.cross_entropy(logits, target)
        triplet_loss, _ = self.triplet(features, target)
        loss = cross_entropy_loss + triplet_loss
        
        return loss
# ------------------------------------

# ------------------------------------
# Trick 6: Center Loss
class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        
        # Select centers for each sample
        centers_batch = self.centers[labels]
        
        # Compute center loss
        loss = torch.sum(torch.pow(features - centers_batch, 2)) / 2.0 / batch_size
        
        # Update centers
        diff = centers_batch - features
        centers_update = torch.zeros_like(self.centers)
        centers_update.scatter_add_(0, labels.unsqueeze(1).repeat(1, self.feat_dim), diff)
        centers_count = torch.zeros_like(self.centers)
        centers_count.scatter_add_(0, labels.unsqueeze(1).repeat(1, self.feat_dim), torch.ones_like(features))
        centers_update /= (centers_count + 1)
        self.centers.data -= centers_update
        
        return loss

class CETCTLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, margin=0.3, e=0, beta=0.0005):
        super(CETCTLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.epsilon = e
        self.beta = beta

        if self.epsilon == 0:
            self.cross_entropy = CrossEntropyLoss()
        else:
            self.cross_entropy = IDLoss(num_classes=self.num_classes, epsilon=self.epsilon)

        self.triplet = TripletLoss(margin=self.margin)
        self.center_loss = CenterLoss(num_classes=self.num_classes, feat_dim=self.feat_dim)

    def forward(self, features, logits, target):
        cross_entropy_loss = self.cross_entropy(logits, target)
        triplet_loss, _ = self.triplet(features, target)
        center_loss = self.center_loss(features, target)
        loss = cross_entropy_loss + triplet_loss + self.beta * center_loss

        return loss
# ------------------------------------