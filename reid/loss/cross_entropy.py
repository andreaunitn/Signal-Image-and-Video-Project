from triplet import TripletLoss
import torch

# ------------------------------------
# Trick 3: Label Smoothing

class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, e):
        log_softmax = torch.nn.functional.log_softmax(input, dim=1)
        loss = -log_softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        mask = target == input.argmax(dim=1)
        n = len(input)
        loss[mask] *= (1-((n-1)/n)*e)
        loss[~mask] *= e/n
        return loss.sum
    
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, margin=1.0, e=0.1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.e = e
        self.cross_entropy_loss = CustomCrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=self.margin)

    def forward(self, input, target):
        cross_entropy_loss = self.cross_entropy_loss(input, target, self.e)
        triplet_loss = self.triplet_loss(input, target)
        loss = cross_entropy_loss + self.alpha * triplet_loss
        return loss
    
# ------------------------------------