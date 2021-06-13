import pytest
import torch
from torch import nn, optim
import os
from transformers import AdamW
from src.models.model import myModel
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# TODO: Setup the test correctly to check for a change of variables when training the model
@pytest.fixture
def setup():
    
    # Import model and optimiser
    model = myModel()
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    # Import data
    train_set = torch.load("../data/processed/train.pt")
    test_set = torch.load("../data/processed/test.pt")

    # Setting up iterator
    train_iter = iter(train_set)
    test_iter = iter(test_set)
    print(len(train_iter.next()))
    batch = [train_iter.next()[0], train_iter.next()[1], train_iter.next()[3]]
    return model, optim, batch


@pytest.fixture
def _train_step(model, optim, batch):
    
    # Put model in training mode
    model.train()

    # Run one forward + backward step 
    optim.zero_grad()
    inputs, attention_masks, labels = batch[0], batch[1], batch[2]

    output = model(inputs, token_type_ids=None, attention_masks=attention_masks, labels=labels)
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

    # Get a list of params that are allowed to change
    if params is None:
        params = [np for np in model.named_parameters() if np[1].requires_grad]
    
    # Take a copy
    initial_params = [(name, p.clone()) for (name, p) in params]

    # Run a training step
    _train_step(model, optim, batch)

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
        
