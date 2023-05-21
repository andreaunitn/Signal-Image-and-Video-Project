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
    def __init__(self, num_classes, alpha=1.0, margin=0.3, e=0):
        super(CETLossV2, self).__init__()
        self.alpha = alpha
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
        #loss = cross_entropy_loss
        
        return loss
    
    # ------------------------------------