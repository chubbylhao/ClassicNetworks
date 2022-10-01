import torch

config = {
    # train and test
    'batch_size': 128,
    'epochs': 3,

    # loss function
    'criterion': 'CrossEntropyLoss',

    # optimizer
    'optimizer': 'SGD',
    'lr': 5e-2,
    'momentum': 0,

    # device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('You are using GPU now!')
    else:
        device = torch.device('cpu')
        print('You are using CPU now!')
