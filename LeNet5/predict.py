import torch
from torch.utils.data import RandomSampler
from data import test_dataset
import matplotlib.pyplot as plt
''' my module '''
from model import LeNet5

if __name__ == '__main__':
    # load model
    lenet5 = LeNet5()
    lenet5.load_state_dict(torch.load('lenet5.pt'))
    lenet5.eval()

    # visualize prediction
    plt.figure('prediction figure')
    plt.suptitle('REAL AND PREDICT')
    rows, cols = 2, 4
    pre_list = list(RandomSampler(test_dataset, num_samples=rows*cols))
    record = []
    for i, index in enumerate(pre_list):
        image, target = test_dataset[index]
        image = torch.reshape(image, (-1, 1, 32, 32))
        pre_target = lenet5(image).argmax(1).item()
        real_target = target
        record.append(1 if pre_target == real_target else 0)
        image = torch.reshape(image, (32, 32))
        plt.subplot(rows, cols, i+1)
        plt.title(f'real: {real_target} pre: {pre_target}')
        plt.imshow(image)
    print(f'The accuracy of the prediction is {sum(record) / len(record) * 100}%')
    plt.show()
