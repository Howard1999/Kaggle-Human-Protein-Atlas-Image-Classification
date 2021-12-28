import PIL.Image as Image
import torch
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset


def _read_image(img_id, img_folder):
    r = cv2.imread(img_folder + '/' + img_id + '_red.png')
    g = cv2.imread(img_folder + '/' + img_id + '_green.png')
    b = cv2.imread(img_folder + '/' + img_id + '_blue.png')
    y = cv2.imread(img_folder + '/' + img_id + '_yellow.png')
    img_ = np.concatenate((r[:, :, 0:1], g[:, :, 0:1], b[:, :, 0:1], y[:, :, 0:1]), axis=2)
    return Image.fromarray(img_)


_TRAINING = 0
_VALIDATION = 1


class MyDataset(Dataset):

    def __init__(self, csv_path, num_classes, img_folder='./dataset', transform=None, mode=_TRAINING, train_ratio=0.85):
        self.img_folder = img_folder
        self.img_list = []
        self.img_label = {}
        self.img_label_multi_hot = []
        self.mode = mode
        self.transform = transform
        self.num_classes = num_classes

        with open(csv_path, newline='') as fp:
            rows = csv.reader(fp)
            for i, row in enumerate(rows):
                if i == 0:
                    continue

                img_id, img_target = row[0], [int(x) for x in row[1].split(' ')]
                self.img_list.append(img_id)
                self.img_label_multi_hot.append(index2multihot(img_target, self.num_classes)[None, :])
                self.img_label[img_id] = img_target

        self.total_size = len(self.img_list)
        self.train_size = int(self.total_size * train_ratio)
        self.validation_size = self.total_size - self.train_size
        self.img_label_multi_hot = torch.cat(self.img_label_multi_hot)

    def __len__(self):
        if self.mode == _TRAINING:
            return self.train_size
        else:
            return self.validation_size

    def __getitem__(self, index):
        if self.mode == _VALIDATION:
            index += self.train_size

        if self.transform is not None:
            return self.transform(_read_image(self.img_list[index], self.img_folder)), \
                   index2multihot(self.img_label[self.img_list[index]], self.num_classes)
        else:
            return _read_image(self.img_list[index], self.img_folder), \
                   index2multihot(self.img_label[self.img_list[index]], self.num_classes)

    def get_multi_hot_labels(self):
        if self.mode == _TRAINING:
            return self.img_label_multi_hot[:self.train_size]
        else:
            return self.img_label_multi_hot[self.train_size:]


def get_train_val_dataset(*args, **kwargs):
    train_dataset = MyDataset(*args, **kwargs, mode=_TRAINING)
    val_dataset = MyDataset(*args, **kwargs, mode=_VALIDATION)
    return train_dataset, val_dataset


def index2multihot(index, num_classes=-1):
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=num_classes).sum(dim=0, dtype=torch.float)


if __name__ == '__main__':
    # test dataset
    dataset = MyDataset('D:/Users/suyih/Downloads/human-protein-atlas-image-classification/train.csv',
                        28,
                        img_folder='D:/Users/suyih/Downloads/human-protein-atlas-image-classification/train',
                        mode=_TRAINING,
                        train_ratio=1.0)
    img, target = dataset[0]
    print(target)
    img.save('./test.png')
