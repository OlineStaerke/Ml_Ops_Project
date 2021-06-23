from random import shuffle
import torch
import pytest
from torch.utils.data import DataLoader, random_split
import os
from transformers import AdamW
from src.data import make_dataset
from src.models.model import myModel
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

class VariablesChangeException(Exception):
    pass

@pytest.fixture
def setup():
    
    # Import model and optimiser
    model = myModel()
    optimizer = AdamW(model.model.parameters(), lr=1e-5, eps=1e-8)
    
    # Import data
    train_dataset = torch.load("../data/raw/train.pt")
    sample_ds = random_split(train_dataset, [30, len(train_dataset) - 30])[0]
    train_dataloader = DataLoader(sample_ds, batch_size=5, drop_last=True)
    # train_dataloader = DataLoader(train_set[10], sampler = train_sampler, batch_size=10)
    return model, optimizer, train_dataloader


def _train_step(model, optim, train_dataloader):
    
    # Put model in training mode
    model.epochs = 1

    # Run one forward + backward step 
    loss = model.train(train_dataloader, with_wandb=False)
    return loss


def test_non_zero_loss(setup):

    model, optimizer, train_dataloader = setup
    
    # Run a training step
    loss = _train_step(model, optimizer, train_dataloader)
    assert not (0 in loss)
        
