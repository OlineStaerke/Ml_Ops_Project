from tqdm import tqdm
import torch
from model import myModel
import os
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
import wandb
from IPython import embed
#embed()
#wandb.init(project="ml_ops_squad")
# Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(
  epochs = 5,
  learning_rate = 1e-5
          ):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  # Change directory
  dir_path = os.path.dirname(os.path.realpath(__file__))
  os.chdir(dir_path)

  #Import data
  train_dataloader = torch.load("../../data/processed/train.pt")
  test_set = torch.load("../../data/processed/test.pt")

  # To train on 5 batches only
  indices = torch.randperm(len(train_dataloader))[:5]
  train_dataset_subset = torch.utils.data.Subset(train_dataloader, indices)

  model = myModel()
  #wandb.watch(model)
  grad_acc_steps = 1
  optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

  train_loss_values = []
  dev_acc_values = []

  for _ in tqdm(range(epochs), desc="Epoch"):

    # Training
    epoch_train_loss = 0 # Cumulative loss
    model.train()
    model.zero_grad()

    # tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable
    # leave ensures one continuous progress bar across all loops
    for step, batch in tqdm(enumerate(train_dataloader),leave=False,total=len(train_dataloader)):

        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)     

        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

        loss = outputs[0]
        loss = loss / grad_acc_steps
        epoch_train_loss += loss.item()

        loss.backward()
        # wandb.log({"loss:": loss})
        
        if (step+1) % grad_acc_steps == 0: # Gradient accumulation is over
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping gradients
          optimizer.step()
          model.zero_grad()
        

    epoch_train_loss = epoch_train_loss / len(train_dataloader)          
    train_loss_values.append(epoch_train_loss)
    
