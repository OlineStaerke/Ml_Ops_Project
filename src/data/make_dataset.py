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
    lookup = {}

    for values in dataloader:
        question = values['question']
        passage = values['passage']
        

        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)
        idx = ' '.join([str(elem) for elem in encoded_pair])
        lookup[idx] = (question, passage)
        

    return np.array(input_ids), np.array(attention_masks), lookup




def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    logger.info('> Loading Dataset')
    train_dataset = load_dataset("boolq",split="train")
    validation_dataset = load_dataset("boolq",split="validation")
    


    logger.info('> Encoding raw data')
    max_seq_length = 256
    input_ids_train, attention_masks_train, _ = encode_data(train_dataset,max_seq_length)
    answers_train = np.array([int(a) for a in train_dataset['answer']])
    input_ids_dev, attention_masks_dev, lookup = encode_data(validation_dataset,max_seq_length)
    answers_dev = np.array([int(a) for a in validation_dataset['answer']])

    train_features = (input_ids_train, attention_masks_train, answers_train)
    dev_features = (input_ids_dev, attention_masks_dev, answers_dev)
    
    # Building Dataloaders
    batch_size = 10
    count = 50
    count_dev = 50
    train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
    dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]

    # for i in range(3):
    #     train_features_tensors[i] = train_features_tensors[i][:count]
    #     dev_features_tensors[i] = dev_features_tensors[i][:count_dev]

    train_dataset = TensorDataset(*train_features_tensors)
    dev_dataset = TensorDataset(*dev_features_tensors)

    #Save raw data
    torch.save(train_dataset,"../../data/raw/train.pt")
    torch.save(dev_dataset,"../../data/raw/test.pt")

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)

    #Save dataloader
    torch.save(lookup, "../../data/processed/test_lookup.pt")
    torch.save(train_dataloader,"../../data/processed/train.pt")
    torch.save(dev_dataloader,"../../data/processed/test.pt")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
    
