# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from datasets import load_dataset
import torch
import os, sys



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    train_dataset = load_dataset("boolq",split="train")

    validation_dataset = load_dataset("boolq",split="validation")
    validation_dataset.set_format('torch')

    torch.save(train_dataset,"../../data/processed/train.pt")
    torch.save(validation_dataset,"../../data/processed/test.pt")

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
