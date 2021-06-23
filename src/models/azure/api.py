import json
from azureml.core import Model
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import os
import joblib


def encode_data(values, max_length):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base") 

    """Encode the question/passage pairs into features than can be fed to the model."""
    input_ids = []
    attention_masks = []


    print(values)
    question = values['question']
    passage = values['passage']
    

    encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
    encoded_pair = encoded_data["input_ids"]
    attention_mask = encoded_data["attention_mask"]

    input_ids.append(encoded_pair)
    attention_masks.append(attention_mask)
    

    return np.array(input_ids), np.array(attention_masks)



def predict(model, input_ids, attention_masks):
    # Evaluation
    model.eval()

    with torch.no_grad():        
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

    print(outputs)
    
    # logits = outputs[0]
    # logits = logits.detach().cpu().numpy()

    # predictions = np.argmax(logits, axis=1).flatten()
    # #labels = labels.numpy().flatten()
    # #print(predictions, attention_masks,labels)
        
    # for num, element in enumerate(sample[0]):
    #     element = element.numpy()
    #     idx = ' '.join([str(elem) for elem in element])
    #     q, p = lookup[idx]
    #     pred = predictions[num]
    #     label = labels[num]
    #     ic(p, q, pred,label)

    return outputs


def init():
    # global model
    # print("This is init")
    # model_path = Model.get_model_path('yesno_model')
    # model = joblib.load(model_path)

    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('./azureml-models/yesno_model/'), 'yesno_model.pkl')
    model = joblib.load(model_path)





def run(data):
    test = json.loads(data)
    print(f"received data {test}")
    # inputs, attentionmask = encode_data(test,256)
    # outputs = predict(model,inputs, attentionmask)
    # return outputs