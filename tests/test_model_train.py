import pytest
import torch
from torch import nn, optim
import os
from transformers import AdamW
from src.models.model import myModel
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

class VariablesChangeException(Exception):
    pass


# TODO: Setup the test correctly to check for a change of variables when training the model
@pytest.fixture
def setup():
    
    # Import model and optimiser
    model = myModel()
    optimizer = AdamW(model.model.parameters(), lr=1e-5, eps=1e-8)

    # Import data
    train_set = torch.load("../data/processed/train.pt")
    test_set = torch.load("../data/processed/test.pt")

    # Setting up iterator
    train_iter = iter(train_set)
    test_iter = iter(test_set)
    next_train_iter = train_iter.next()
    batch = [next_train_iter[0], next_train_iter[1], next_train_iter[2]]
    return model, optimizer, batch


def _train_step(model, optim, batch):
    
    # Put model in training mode
    model.epochs = 1
    model.model.train()

    # Run one forward + backward step 
    optim.zero_grad()
    inputs, attention_masks, labels = batch[0], batch[1], batch[2]

    output = model.model(inputs, token_type_ids=None, attention_mask=attention_masks, labels=labels)
    loss = output[0]
    loss.backward()
    optim.step()

@pytest.fixture
def _forward_step(model, batch):

    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        # inputs and targets
        inputs = batch[0]
        # move data to DEVICE
        inputs = inputs
        # forward
        return model(inputs)

def test_var_change(setup):

    model, optimizer, batch = setup
    
    # Take a copy
    initial_params = [ np for np in model.model.named_parameters() if np[1].requires_grad ]
    # Run a training step
    _train_step(model, optimizer, batch)

    params = [ np for np in model.model.named_parameters() if np[1].requires_grad ]

    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            raise VariablesChangeException(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg = 'did not change'
                )
            )
        
