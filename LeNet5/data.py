import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
''' my module '''
from h_paras import config

# data augmentation
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

# get MNIST dataset
train_dataset = datasets.MNIST('./', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('./', train=False, transform=transform, download=True)

# get iterable
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4*config['batch_size'], shuffle=False)

if __name__ == '__main__':
    image, target = train_dataset[0]
    print(f'The image size is {image.shape}, the image label is {target}')
    image = torch.reshape(image, (32, 32))
    plt.figure()
    plt.imshow(image)
    plt.show()
