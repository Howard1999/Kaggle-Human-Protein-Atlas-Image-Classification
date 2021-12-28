import torch 


def inference(model, image, device='cpu', threshold=0.5, to_label=True):
    y_pred = torch.sigmoid(model(image[None, :].to(device)))
    if to_label:
        if type(threshold)==float:
            return [x[1] for x in (y_pred > threshold).nonzero(as_tuple=False).tolist()]
        elif type(threshold)==list:
            return [x[1] for x in (y_pred > torch.tensor([threshold], device=device)).nonzero(as_tuple=False).tolist()]
        else:
            raise Exception('unknown threshold type')
    else:
        return y_pred
