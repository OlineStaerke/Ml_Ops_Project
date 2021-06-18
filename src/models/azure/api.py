import json
from azureml.core import Model
import joblib


def init():
    global model
    print("This is init")
    model_path = Model.get_model_path('yesno_model')
    model = joblib.load(model_path)



def run(data):
    test = json.loads(data)
    print(f"received data {test}")
    return f"test is {test}"