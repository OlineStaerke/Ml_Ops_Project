from tqdm import tqdm
import torch
from model import myModel
import os, shutil
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
#import wandb
#from IPython import embed

#################
#HYPERPARAMETERS#
#################

# Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_acc_steps = 1
model = myModel(epochs, learning_rate, grad_acc_steps, device)

run.log('Epochs', epochs)
run.log('Learning rate', learning_rate)
run.log('Gradient accumulation steps', grad_acc_steps)
###############
#DATA LOADING##
###############

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
run.log('os', os.getcwd())
#Import data
train_dataloader = torch.load("../../data/processed/train.pt")
test_set = torch.load("../../data/processed/test.pt")

# TODO: This is only if you wish to work with 5 batches at a time
# To train on 5 batches only
indices = torch.randperm(len(train_dataloader))[:5]
train_dataset_subset = torch.utils.data.Subset(train_dataloader, indices)

##################
#WEIGHTS & BIASES#
##################

#wandb.watch(model)
#wandb.init(project="ml_ops_squad")

##########
#TRAINING#
##########

train_loss = model.train(train_dataloader)
run.log('Training loss', train_loss)

########
#SAVING#
########

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
torch.save(model.model, "../../models/model.pth")

run.complete()

