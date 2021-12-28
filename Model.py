import timm
import torch


def get_model(class_size, checkpoint=None):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 3, (1, 1)),
        timm.create_model('resnet50', num_classes=class_size, pretrained=True)
    )
    
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    
    return model
