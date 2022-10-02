import torch
from torch.utils.data import RandomSampler
from data import test_dataset, labels
import matplotlib.pyplot as plt
import numpy as np
''' my module '''
from model import AlexNet

if __name__ == '__main__':
    # load model
    alexnet = AlexNet()
    alexnet.load_state_dict(torch.load('alexnet.pt'))
    alexnet.eval()

    # visualize prediction
    plt.figure('prediction figure')
    plt.suptitle('REAL AND PREDICT')
    rows, cols = 2, 4
    pre_list = list(RandomSampler(test_dataset, num_samples=rows*cols))
    record = []
    for i, index in enumerate(pre_list):
        image, target = test_dataset[index]
        image = torch.reshape(image, (-1, 3, 224, 224))
        pre_target = alexnet(image).argmax(1).item()
        real_target = target
        record.append(1 if pre_target == real_target else 0)
        image = torch.reshape(image, (3, 224, 224))
        ''' FOR DISPLAY '''
        # scale the image range to [0, 1] (unnormalize)
        image = (image + 1) / 2
        # transform [C, H, W] to [H, W, C] for display
        image = np.transpose(image.numpy(), (1, 2, 0))
        plt.subplot(rows, cols, i + 1)
        plt.title(f'real: {labels[real_target]} pre: {labels[pre_target]}', fontdict={'fontsize': 10})
        plt.imshow(image)
    print(f'The accuracy of the prediction is {sum(record) / len(record) * 100}%')
    plt.show()
