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
    def __init__(self, num_classes, feat_dim, device="cuda"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = device

        self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        batch_size = x.size(0)
        features = x.view(batch_size, -1)

        # compute center loss
        centers_batch = self.centers[labels, :]
        center_loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size

        # update centers
        diff = centers_batch - features
        unique_label, unique_idx = torch.unique(labels, return_inverse=True)
        appear_times = torch.histc(unique_idx.float(), bins=self.num_classes, min=0, max=self.num_classes-1)
        appear_times = appear_times.to(self.device)
        diff_cpu = diff.data.cpu()
        for i in range(self.num_classes):
            if appear_times[i] == 0:
                continue
            else:
                self.centers[i] -= torch.mean(diff_cpu[labels == i, :], dim=0) * 0.5

        return center_loss

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