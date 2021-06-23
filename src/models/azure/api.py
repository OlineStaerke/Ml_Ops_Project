import json
from azureml.core import Model
import joblib

def predict(model, input_ids,attention_masks):
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


def init():
    global model
    print("This is init")
    model_path = Model.get_model_path('yesno_model')
    model = joblib.load(model_path)



def run(data):
    test = json.loads(data)
    print(f"received data {test}")
    return f"test is {test}"