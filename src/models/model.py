import torch
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
from tqdm import tqdm
import torchdrift


class myModel():
    # Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self,
                    epochs = 5,
                    lr = 1e-5,
                    grad_acc_steps = 1,
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ):
        # Set seeds for reproducibility
        random.seed(26)
        np.random.seed(26)
        torch.manual_seed(26)

        self.epochs = epochs
        self.lr = lr
        self.grad_acc_steps = grad_acc_steps
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        self.device = device
        self.model.to(self.device) # Send the model to the GPU if we have one
  

    def train(self, train_dataloader, eval_dataloader):

        train_loss_values = []
        eval_loss_values = []
        dev_acc_values = []
        drift_detection = []

        #Drift detector
        drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
       

        for _ in tqdm(range(self.epochs), desc="Epoch"):

            # Training
            epoch_train_loss = 0 # Cumulative loss
            epoch_eval_loss = 0

            self.model.train()
            self.model.zero_grad()

            # tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable
            # leave ensures one continuous progress bar across all loops
            for step, batch in tqdm(enumerate(train_dataloader),leave=False,total=len(train_dataloader)):

                input_ids = batch[0].to(self.device)
                attention_masks = batch[1].to(self.device)
                labels = batch[2].to(self.device)     

                outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

                loss = outputs[0]
                loss = loss / self.grad_acc_steps
                epoch_train_loss += loss.item()

                loss.backward()
                # wandb.log({"loss:": loss})
                
                if (step+1) % self.grad_acc_steps == 0: # Gradient accumulation is over
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clipping gradients
                    self.optimizer.step()
                    self.model.zero_grad()

                

            epoch_train_loss = epoch_train_loss / len(train_dataloader)          
            train_loss_values.append(epoch_train_loss)

            #Evaluate
            for step, batch in tqdm(enumerate(eval_dataloader),leave=False,total=len(eval_dataloader)):
                with torch.no_grad():
                    self.model.eval()
                    input_ids = batch[0].to(self.device)
                    attention_masks = batch[1].to(self.device)
                    labels = batch[2].to(self.device)     

                    outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

                    loss = outputs[0]
                    loss = loss/self.grad_acc_steps

                    epoch_eval_loss += loss.item()

                    #DRIFT DETECTION
                    # score = drift_detector(outputs)
                    # p_val = drift_detector.compute_p_value(outputs)
                    # drift_detection.append([(score,p_val)])

            epoch_eval_loss = epoch_eval_loss / len(eval_dataloader)          
            eval_loss_values.append(epoch_eval_loss)



        return train_loss_values, eval_loss_values, drift_detection

        
