import torch
import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, 5)),
            # ('sigmoid1', nn.Sigmoid()),
            ('relu1', nn.ReLU()),
            ('pooling1', nn.AvgPool2d(2)),
            ('conv2', nn.Conv2d(6, 16, 5)),
            # ('sigmoid2', nn.Sigmoid()),
            ('relu2', nn.ReLU()),
            ('pooling2', nn.AvgPool2d(2)),
            # input to the FCN is a vector
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(16*5*5, 120)),
            # ('sigmoid3', nn.Sigmoid()),
            ('relu3', nn.ReLU()),
            ('linear2', nn.Linear(120, 84)),
            # ('Sigmoid4', nn.Sigmoid()),
            ('relu4', nn.ReLU()),
            ('linear3', nn.Linear(84, 10))
        ]))

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    x = torch.ones((1, 32, 32), dtype=torch.float32)
    x = torch.reshape(x, (-1, 1, 32, 32))
    lenet5 = LeNet5()
    print(f'The model : \n {lenet5}')
    print(f'The shape of LeNet5\'s output is {lenet5(x).shape}')
