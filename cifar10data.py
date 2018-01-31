import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import calc_dataset_stats
import numpy as np


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
                                     shuffle=False, num_workers=args.dataloader_workers, pin_memory=args.pin_memory)

    def plot_random_sample(self):
        # Get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()
        print(images[0])
        exit(1)
        # Show images
        grid = torchvision.utils.make_grid(images)
        img = grid / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        # Print labels
        print(' '.join('%5s' % CIFAR10_LABELS_LIST[labels[j]] for j in range(len(labels))))


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
