from .triplet import TripletLoss
import torch

# ------------------------------------
# Trick 3: Label Smoothing
    
class CETLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, margin=1.0, e=0): #use e=0.1 for LabelSmoothing
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.e = e
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing = e) #TODO move to GPU
        self.triplet_loss = TripletLoss(margin=self.margin) #TODO move to GPU

    def forward(self, features, logits, target):
        cross_entropy_loss = self.cross_entropy_loss(logits, target)
        triplet_loss, _ = self.triplet_loss(features, target)
        loss = cross_entropy_loss + self.alpha * triplet_loss
        return loss
    
# ------------------------------------