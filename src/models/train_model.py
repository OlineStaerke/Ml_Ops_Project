from tqdm import tqdm
import torch
from model import myModel
import os, shutil
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
from azureml.core import Run, Workspace, Dataset
import joblib
import io
from create_dataloader import data_loader

run = Run.get_context()
ws = Workspace.from_config()

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

#Get datastore default
datastore = ws.get_default_datastore()

# create a dataset referencing the cloud location
dataset_train = Dataset.Tabular.from_json_lines_files(path = [(datastore, ('data/train.json'))])
dataset_test = Dataset.Tabular.from_json_lines_files(path = [(datastore, ('data/val.json'))])

#Create dataloader objects, from pandas dataframe.
train_dataloader, test_dataloader = data_loader(dataset_train.to_pandas_dataframe(),dataset_test.to_pandas_dataframe())

##########
#TRAINING#
##########

train_loss, eval_loss, drift_detection = model.train(train_dataloader, test_dataloader)
run.log_list('TrainingLoss', train_loss)
run.log_list('Eval loss', eval_loss)
run.log_list('Drift Detection', drift_detection)

########
#SAVING#
########

# Save the trained model in Azure
model_file = 'yesno_model.pkl'
joblib.dump(value=model.model, filename=model_file)
run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

#complete run
run.complete()

# Register the model
run.register_model(model_path='outputs/yesno_model.pkl', model_name='yesno_model',
                   tags={'Training context':'Inline Training'},
                   properties={'AUC': run.get_metrics()['TrainingLoss'], 'TrainingLoss': run.get_metrics()['TrainingLoss']})

print('Model trained and registered.')
