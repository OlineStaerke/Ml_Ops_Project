import torch, os
from IPython import embed
import numpy as np
from icecream import ic
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

test_dir = "../../data/processed/test.pt"
test_dataloader = torch.load(test_dir)
test_dir = "../../data/processed/test_lookup.pt"
lookup = torch.load(test_dir)
sample = iter(test_dataloader).next()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = torch.load("../../models/model.pth")

# Evaluation
#model.eval()
    
input_ids = sample[0]
attention_masks = sample[1]
labels = sample[2]
print("###############")
print(input_ids)
print(attention_masks)

with torch.no_grad():        
    outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

logits = outputs[0]
logits = logits.detach().cpu().numpy()

predictions = np.argmax(logits, axis=1).flatten()
#labels = labels.numpy().flatten()
#print(predictions, attention_masks,labels)

for num, element in enumerate(sample[0]):
  element = element.numpy()
  idx = ' '.join([str(elem) for elem in element])
  q, p = lookup[idx]
  pred = predictions[num]
  label = labels[num]
  ic(p, q, pred,label)

