import torch
import Dataset
import Augmentation
import Optimizer
import Model
import Loss
import Validation
from torch.utils.data import DataLoader

# hyper parameter
TOTAL_EPOCH = 100
LEARNING_RATE = 1e-6
BATCH_SIZE = 2
CLASSES_SIZE = 28
device = torch.device('cuda')

# initial dataset
transform = Augmentation.get_transform()
dataset_path = 'D:/Users/suyih/Downloads/human-protein-atlas-image-classification/'
train_dataset, validation_dataset = Dataset.get_train_val_dataset(dataset_path + 'train.csv', CLASSES_SIZE,
                                                                  img_folder=dataset_path + 'train/',
                                                                  transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# initial model
model = Model.get_model(CLASSES_SIZE)
model.to(device)

# initial optimizer and scheduler
optimizer = Optimizer.get_optimizer(model, LEARNING_RATE)
scheduler = Optimizer.get_scheduler(optimizer)

# loss
loss_func = Loss.BCELoss()

# training loop
model.train()
for epoch in range(TOTAL_EPOCH):
    # train
    for images, targets in train_data_loader:
        y_pred = model(images.to(device))
        loss = loss_func(y_pred, targets, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    # validation
    val_loss, metrics = Validation.validation(model, validation_data_loader, loss_func, device)
