# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from datasets import load_dataset
import torch
import os, sys
from tqdm import tqdm
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np

def encode_data(dataloader, max_length):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base") 

    """Encode the question/passage pairs into features than can be fed to the model."""
    input_ids = []
    attention_masks = []
    answers = []
   
    

    for values in dataloader:
        answer = values['answer']
        question = values['question']
        passage = values['passage']
        
        answers.append(answer)

        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)
        

    return np.array(input_ids), np.array(attention_masks), answers




def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    train_dataset = load_dataset("boolq",split="train")
    validation_dataset = load_dataset("boolq",split="validation")
    
  
    #Save not encoded data
    #torch.save(train_dataset,"../../data/raw/train.pt")
    #torch.save(validation_dataset,"../../data/raw/test.pt")

    max_seq_length = 256
    input_ids_train, attention_masks_train, answers_train = encode_data(train_dataset,max_seq_length)
    input_ids_dev, attention_masks_dev, answers_dev = encode_data(validation_dataset,max_seq_length)

    train_features = (input_ids_train, attention_masks_train, answers_train)
    dev_features = (input_ids_dev, attention_masks_dev, answers_dev)
    

    #Creating the dataloader

    # Building Dataloaders
    batch_size = 32

    train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
    dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]

    train_dataset = TensorDataset(*train_features_tensors)
    dev_dataset = TensorDataset(*dev_features_tensors)

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)

    #Save dataloader
    torch.save(train_dataloader,"../../data/processed/train.pt")
    torch.save(dev_dataloader,"../../data/processed/test.pt")

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
