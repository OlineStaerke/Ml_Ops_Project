import torch
from IPython import embed

def predict():
  
  test_dir = "../../data/processed/test.pt"
  test_dataloader = torch.load(test_dir)
  embed()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = torch.load(model.model, "../../models/model.pth")
  
  # Evaluation
  model.eval()
      
  input_ids = sample[0].to(device)
  attention_masks = sample[1].to(device)
  labels = sample[2]

  with torch.no_grad():        
      outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()

  predictions = np.argmax(logits, axis=1).flatten()
  labels = labels.numpy().flatten()

  return labels

predict()
        
