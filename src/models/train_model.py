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
import hydra
from omegaconf import DictConfig
import logging

#################
#HYPERPARAMETERS#
#################
@hydra.main(config_path="../../", config_name = "config.yaml")
def my_app(cfg: DictConfig) -> None:
    #Log in hydra the params
    log = logging.getLogger(__name__)
    log.info(cfg)
    

    #wandb.init(project="ml_ops_squad")
    # Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = cfg.epochs
    learning_rate = cfg.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_acc_steps = cfg.grad_acc_steps

    model = myModel(epochs, learning_rate, grad_acc_steps, device)

    ###############
    #DATA LOADING##
    ###############

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    #Import data
    print("Current directory:") #Current dorectory changes to log files when using hydra
    print(os.getcwd())
    
    train_dataloader = torch.load("../../../../../data/processed/train.pt")
    test_set = torch.load("../../../../../data/processed/test.pt")

    ##################
    #WEIGHTS & BIASES#
    ##################

    #wandb.watch(model.model)

    ##########
    #TRAINING#
    ##########

    train_loss = model.train(train_dataloader)
    ########
    #SAVING#
    ########

    torch.save(model.model, "../../models/model.pth")


if __name__ == "__main__":
    my_app()
