import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
''' my module '''
from h_paras import config

# data augmentation
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# get MNIST dataset
train_dataset = datasets.CIFAR10('./', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10('./', train=False, transform=transform, download=True)

# get iterable
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# targets and labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    image, target = train_dataset[0]
    print(f'The image size is {image.shape}, the image label is {labels[target]}')
    image = np.transpose(image, (1, 2, 0))
    plt.figure()
    plt.imshow(image)
    plt.show()
