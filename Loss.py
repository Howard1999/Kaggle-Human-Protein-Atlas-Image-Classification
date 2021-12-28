import torch


class BCELoss:
    def __init__(self, pos_weight=None):
        if pos_weight is not None:
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()

    # target is index
    def __call__(self, pred, true, device='cpu'):
        return self.loss(pred, true.to(device))

    
def focal_BCELoss(logits, targets, device='cpu', gamma=2, num_label=28):
    logits.to(device)
    targets.to(device)
    
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where((t >= 0.5).to(device), p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = num_label*loss.mean()
    return loss
    
    