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
import optuna
OPTUNA=True
wandb.init("Ml_Ops_Squad")
###############
#DATA LOADING##
###############
def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    #Import data
    print("#######")
    print("Current directory:") #Current dorectory changes to log files when using hydra
    print(os.getcwd())

    if not OPTUNA:
        train_dir = "../../../../../data/processed/train.pt"
        test_dir = "../../../../../data/processed/test.pt"
    else:
        train_dir = "../../data/processed/train.pt"
        test_dir = "../../data/processed/test.pt"

    train_dataloader = torch.load(train_dir)
    test_dataloader = torch.load(test_dir)

    return train_dataloader,test_dataloader

########
#OPTUNA#
########
def my_model_optuna(trial):
    #Get paramters from OPTUNA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = trial.suggest_loguniform('lr',0.0001,0.1)
    epochs = int(trial.suggest_discrete_uniform('epochs',5,25,1))
    optimizer =  trial.suggest_categorical('optimizer',[torch.optim.SGD, torch.optim.RMSprop, torch.optim.AdamW])
    grad_acc_steps = 1
    model = myModel(epochs, learning_rate, grad_acc_steps, device, optimizer)
    wandb.watch(model.model)
    return model
    
def objective(trial):
    model = my_model_optuna(trial)
    train_dataloader, val_dataloader = load_data() #load data
    return(train_model(model, train_dataloader, val_dataloader)) #train model, returns val accuracy


#################
#HYPERPARAMETERS#
#################
@hydra.main(config_path="../../", config_name = "config.yaml")
def my_model_hydra(cfg: DictConfig) -> None:
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

    model = myModel(epochs, learning_rate, grad_acc_steps, device) #init model
    wandb.watch(model.model)
    train_dataloader, val_dataloader = load_data() #load data
    train_model(model, train_dataloader, val_dataloader) #train model

##########
#TRAINING#
##########
def train_model(model, train_dataloader, val_dataloader):
    
    ##################
    #WEIGHTS & BIASES#
    ##################

    

    #Train the model
    train_loss, val_acc = model.train(train_dataloader, val_dataloader )

    ########
    #SAVING#
    ########

    torch.save(model.model, "../../models/model.pth")

    return val_acc[-1]



if __name__ == "__main__":
    if not OPTUNA:
        model = my_model_hydra()
        wandb.watch(model)
    
    else:
        study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        study.optimize(objective, n_trials=20) 

        

