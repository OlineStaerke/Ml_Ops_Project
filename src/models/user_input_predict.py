
import torch, os
from IPython import embed
import numpy as np
from icecream import ic
from transformers import AutoTokenizer

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../../models/model.pth")

# Evaluation
model.eval()
passage = str(input("Type passage: "))
question = str(input("Type question: "))

encoded_data = tokenizer.encode_plus(
            question,
            passage,
            max_length=256,
            pad_to_max_length=True,
            truncation_strategy="longest_first",
        )
encoded_pair = torch.Tensor(encoded_data["input_ids"])
attention_mask = torch.Tensor(encoded_data["attention_mask"])


input_ids = encoded_pair.to(device)
attention_masks = attention_mask.to(device)
embed()
with torch.no_grad():
    outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

logits = outputs[0]
logits = logits.detach().cpu().numpy()

predictions = np.argmax(logits, axis=1).flatten()
# labels = labels.numpy().flatten()
# print(predictions, attention_masks,labels)

ic(predictions)
