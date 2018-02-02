import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import calc_dataset_stats


# Example DataLoader on CIFAR-10

class CIFAR10Data:
    def __init__(self, args):
        mean, std = calc_dataset_stats(torchvision.datasets.CIFAR10(root='./data', train=True,
                                                                    download=args.download_dataset).train_data,
                                       axis=(0, 1, 2))

        train_transform = transforms.Compose(
            [transforms.RandomCrop(args.img_height),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.3, 0.3, 0.3),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

        self.trainloader = DataLoader(torchvision.datasets.CIFAR10(root='./data', train=True,
                                                                   download=args.download_dataset,
                                                                   transform=train_transform),
                                      batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.dataloader_workers,
                                      pin_memory=args.pin_memory)

        self.testloader = DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False,
                                                                  download=args.download_dataset,
                                                                  transform=test_transform),
                                     batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.dataloader_workers,
                                     pin_memory=args.pin_memory)


CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
