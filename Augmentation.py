from torchvision import transforms


def get_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(360),
        transforms.ToTensor(),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from Dataset import *

    dataset = MyDataset('../datasets/training_images', transform=get_transform())

    label, img = dataset[5]
    print(img.shape)
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
