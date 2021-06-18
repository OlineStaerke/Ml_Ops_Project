import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json

def encode_data(dataloader, max_length):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base") 

    """Encode the question/passage pairs into features than can be fed to the model."""
    input_ids = []
    attention_masks = []
    print("#####################")
    

    for index, values in dataloader.iterrows():

        question = values['question']
        passage = values['passage']
        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)
        

    return np.array(input_ids), np.array(attention_masks)


def data_loader(train_dataset, validation_dataset):
    max_seq_length = 256
    input_ids_train, attention_masks_train = encode_data(train_dataset,max_seq_length)
    answers_train = np.array([int(a) for a in train_dataset['answer']])
    input_ids_dev, attention_masks_dev = encode_data(validation_dataset,max_seq_length)
    answers_dev = np.array([int(a) for a in validation_dataset['answer']])

    train_features = (input_ids_train, attention_masks_train, answers_train)
    dev_features = (input_ids_dev, attention_masks_dev, answers_dev)
    

    # Building Dataloaders
    batch_size = 10
    count = 50
    count_dev = 50
    train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
    dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]


    train_dataset = TensorDataset(*train_features_tensors)
    dev_dataset = TensorDataset(*dev_features_tensors)

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)
    
    return train_dataloader,dev_dataloader

