import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Example DataLoader on CIFAR-100

class CIFAR100Data:
    def __init__(self, args):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.trainloader = DataLoader(torchvision.datasets.CIFAR100(root='./data', train=True,
                                                                    download=args.download_dataset,
                                                                    transform=transform),
                                      batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.dataloader_workers,
                                      pin_memory=args.pin_memory)

        self.testloader = DataLoader(torchvision.datasets.CIFAR100(root='./data', train=False,
                                                                   download=args.download_dataset, transform=transform),
                                     batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.dataloader_workers, pin_memory=args.pin_memory)

    def plot_random_sample(self):
        # Get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        # Show images
        grid = torchvision.utils.make_grid(images)
        img = grid / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        # Print labels
        print(' '.join('%5s' % CIFAR100_LABELS_LIST[labels[j]] for j in range(len(labels))))


CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
