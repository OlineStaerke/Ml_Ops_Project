import pytest
import torch
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

def test_data_size():

    trainloader = torch.load("../data/processed/train.pt")
    testloader = torch.load("../data/processed/test.pt")

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