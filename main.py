import time
import math
import numpy as np

import torchvision
import torch
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.model import CCT
from util.auto_augment import CIFAR10Policy
from util.loss import LabelSmoothingCrossEntropy

from tensorboardX import SummaryWriter

writer = SummaryWriter()


def adjust_lr(optimizer, epoch, warm_up=5, lr=5e-4, total_epoch=200):
    new_lr = lr
    if epoch < warm_up:
        new_lr = lr / (warm_up - epoch)
    else:
        new_lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warm_up) / (total_epoch - warm_up)))
    for param in optimizer.param_groups:
        param['lr'] = new_lr


def train(model, epoch, train_loader, optimizer, criterion, device, print_freq=10):
    total_acc = 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        n += images.size(0)  # 这个epoch下已经读取了多少图片

        optimizer.zero_grad()
        pred = model(images)  # [batch, 10]
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        acc = accuracy(pred, target)  # 这个batch下预测对了多少
        total_acc += acc
        avg_acc = total_acc * 100.0 / n

        if i % print_freq == 0:
            writer.add_scalar('NLL loss with label smoothing', loss.item(), global_step=i + epoch * 391 + 1)
            writer.add_scalar('Train dataset top-1', avg_acc, global_step=i + epoch * 391 + 1)


def validate(model, epoch, val_loader, criterion, device, print_freq=200):
    model.eval()

    n = 0
    total_acc = 0.0
    avg_acc = 0.0
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)
        n += images.size(0)

        pred = model(images)  # [batch, 10]
        loss = criterion(pred, target)
        loss.backward()

        acc = accuracy(pred, target)  # 这个batch下预测对了多少
        total_acc += acc
        avg_acc = total_acc * 100.0 / n

        if i % print_freq == 0:
            print("validate epoch %d iter %d top1 %f" % (epoch + 1, i + 1, avg_acc))

    print("validate ends avg_acc %f" % avg_acc)
    writer.add_scalar('Valid dataset top-1', avg_acc, global_step=epoch + 1)

    model.train()


def accuracy(pred, target):
    _, res = F.softmax(pred, dim=-1).topk(1, 1, True, True)
    res = res.squeeze(-1)
    acc = (res == target).numpy().astype(np.uint8).sum()
    return acc


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    model = CCT().to(device)
    criterion = LabelSmoothingCrossEntropy()
    optimizer = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=3e-2)

    train_transform = transforms.Compose([CIFAR10Policy,
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)
                                          ])
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
    train_dataset = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataset = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=False, transform=valid_transform)
    val_loader = DataLoader(val_dataset)

    total_epoch = 200

    for epoch in range(total_epoch):
        print("epoch %d starts" % (epoch + 1))
        adjust_lr(optimizer, epoch)
        time_before = time.time()
        train(model, epoch, train_loader, optimizer, criterion, device)
        time_after = time.time()
        print("epoch %d ends, time: %f minutes" % (epoch + 1, (time_after - time_before) / 60.0))
        validate(model, epoch, val_loader, criterion, device)

    print("task ends")


if __name__ == 'main':
    main()
