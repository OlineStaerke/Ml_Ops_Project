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

    # Question size
    assert train_iter.next()[0].shape and test_iter.next()[0].shape == torch.Size([10, 256])
    # Passage size
    assert train_iter.next()[1].shape and test_iter.next()[1].shape == torch.Size([10, 256])
    # Answer size
    assert train_iter.next()[2].shape and test_iter.next()[2].shape == torch.Size([10])