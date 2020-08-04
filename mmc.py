import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import optim
from adv_package.loader import ModelManager
from adv_package.utils import EpochTracker
from easydict import EasyDict
import yaml, os

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = EasyDict(yaml.safe_load(file))

    return config

# hyperparameters
model = ModelManager(load_config('test.yaml').THREAT_MODEL)
model = model.getModel()
iterations = 200
lr = 0.01
os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'


train_data = CIFAR10('datasets/', train=True, transform=transforms.ToTensor(), download=False)
batch_loader = DataLoader(train_data, batch_size=64)

optim = optim.SGD(model.parameters,
                  lr=0.01,
                  momentum=0.9)


for itr in range(iterations):
    tracker = EpochTracker(iterations)
    tracker.setEpoch(itr)

    for i, (x,y) in enumerate(batch_loader):
        optim.zero_grad()
        x = x.cuda()
        y = y.cuda()

        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()

        tracker.store(logits, loss, y)

        optim.step()


        if i % 10 == 0:
            print(tracker.log(i, len(batch_loader), True))


