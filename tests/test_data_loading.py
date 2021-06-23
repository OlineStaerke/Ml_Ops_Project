import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

def test_data_size():

    train_dataset = torch.load("../data/raw/train.pt")
    dev_dataset = torch.load("../data/raw/test.pt")

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)

    trainloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=10)
    testloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=10)


    train_iter = iter(trainloader)
    test_iter = iter(testloader)

    assert len(trainloader) == 943
    assert len(testloader) == 327
    
    next_train_iteration, next_test_iteration = train_iter.next(), test_iter.next()
    
    # Question size
    assert next_train_iteration[0].shape and next_test_iteration[0].shape == torch.Size([10, 256])
    # Passage size
    assert next_train_iteration[1].shape and next_test_iteration[1].shape == torch.Size([10, 256])
    # Answer size
    assert next_train_iteration[2].shape and next_test_iteration[2].shape == torch.Size([10])