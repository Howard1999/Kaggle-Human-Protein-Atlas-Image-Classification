import torch
import Metrics


def validation(model, dataloader, loss_function, device='cpu'):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in dataloader:
            y_true.append(label)
            y_pred.append(model(img.to(device)))
        y_pred, y_true = torch.cat(y_pred), torch.cat(y_true).to(device)
        loss = loss_function(y_pred, y_true, device)
        metrics = Metrics.metrics(y_pred, y_true)
    model.train()
    return loss.item(), metrics
