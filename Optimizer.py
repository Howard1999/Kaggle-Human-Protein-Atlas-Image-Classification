from torch import optim


def get_optimizer(model, lr=1e-5):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


def get_scheduler(optimizer, epoch=50):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
