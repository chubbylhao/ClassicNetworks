import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
''' my module '''
from data import test_dataset, train_dataloader, test_dataloader
from h_paras import config
from model import AlexNet

# build model
alexnet = AlexNet().to(config['device'])

# loss function
criterion = getattr(nn, config['criterion'])().to(config['device'])

# optimizer
optimizer = getattr(optim, config['optimizer'])(
    alexnet.parameters(), config['lr'])

# record loss for each update
record = {
    'train_loss': [],
    'test_loss': [],
    'pos_examples': [],
}

# start training!
print('\n----------training start----------\n')
_, ax = plt.subplots()
alexnet.train()
for epoch in range(config['epochs']):
    for index, (images, targets) in enumerate(train_dataloader):
        # if you want a larger batch size but limited by memory, set this code after a few iterations
        optimizer.zero_grad()
        fp_res = alexnet(images.to(config['device']))
        loss = criterion(fp_res, targets.to(config['device']))
        loss.backward()
        optimizer.step()
        record['train_loss'].append(loss.item())
        if index % 50 == 0:
            plt.cla()
            ax.plot(list(range(len(record['train_loss']))),
                    record['train_loss'])
            plt.xlabel('update times')
            plt.ylabel('loss')
            plt.pause(0.1)
    print(f'epoch: {epoch + 1}, loss: {record["train_loss"][-1]}')
print('\n----------training done----------\n')
# plt.pause(3)
plt.show()

# start testing!
print('\n----------testing start----------\n')
alexnet.eval()
with torch.no_grad():
    for images, targets in test_dataloader:
        fp_res = alexnet(images.to(config['device']))
        loss = criterion(fp_res, targets.to(config['device']))
        record['test_loss'].append(loss.item())
        record['pos_examples'].append(sum((fp_res.detach().cpu().argmax(1) == targets)))
print(f'total loss is {sum(record["test_loss"])}')
print(f'accuracy is {sum(record["pos_examples"]) / len(test_dataset)}')
print('\n----------testing done----------\n')

# save model
torch.save(alexnet.state_dict(), 'alexnet.pt')
