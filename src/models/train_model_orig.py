from tqdm import tqdm
import torch
from model import myModel
import os, shutil
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
import wandb
from IPython import embed

#################
# HYPERPARAMETERS#
#################

wandb.init(project="ml_ops_squad")
# Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_acc_steps = 1
model = myModel(epochs, learning_rate, grad_acc_steps, device)

# run.log('Epochs', epochs)
# run.log('Learning rate', learning_rate)
# run.log('Gradient accumulation steps', grad_acc_steps)
###############
# DATA LOADING##
###############

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# run.log('os', os.getcwd())
# Import data
embed()
train_dataloader = torch.load("../../data/processed/train.pt")
test_set = torch.load("../../data/processed/test.pt")

##################
# WEIGHTS & BIASES#
##################

wandb.watch(model.model)

##########
# TRAINING#
##########

train_loss = model.train(train_dataloader)
# run.log('Training loss', train_loss)

########
# SAVING#
########

torch.save(model.model, "../../models/model.pth")

# run.complete()
