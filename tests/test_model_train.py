from tqdm import tqdm
import torch

import os
import sys
path_cur=os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from src.models.model import myModel
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
import wandb
from IPython import embed
from src.models.train_model import train



os.chdir(path_cur)
def test_model():
    train(epochs=1, learning_rate=1e-5)