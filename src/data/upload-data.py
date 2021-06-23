# upload-data.py
from azureml.core import Workspace, Dataset
import json
from datasets import load_dataset
import pandas as pd

train_dataset = load_dataset("boolq", split="train")
validation_dataset = load_dataset("boolq", split="validation")


ws = Workspace.from_config()


local_path = "../../data/processed/train.json"
train_dataset.to_json(local_path)


local_path = "../../data/processed/val.json"
validation_dataset.to_json(local_path)


# get the datastore to upload prepared data
datastore = ws.get_default_datastore()

# upload the local file from src_dir to the target_path in datastore
datastore.upload(src_dir="../../data/processed/", target_path="data")
