# Import necessary libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import BertTokenizer, BertForSequenceClassification

print("Libraries imported")

class TweetsDataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings['input_ids'])

class MisogynyDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    #print('MISOGYNY')
    #print(labels)
    self.encodings  = encodings
    self.labels = labels
    #print(self.labels)

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  print("classes imported")

"""### Part 1: Pre-train with a large volume of random tweets in Spanish. """

#corpus = pd.read_csv("./test.txt")
text = pd.read_csv(".covidcorpusLarge.csv")   # ./test.txt")
text = text.sample(n=5549696)

text = text['content']
print("Text")

text = text.astype('str')
print("Astype")

text = text.tolist()
#text = text.lower()
print("tolist")

print("spanish corpus imported")

from transformers import BertTokenizer, BertForMaskedLM

# create the tokenizer and load the pre-trained model

tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", do_lower_case=True)
print("Tokenizer imported")
#state_dict=torch.load("/home/dalia/anaconda3/pkgs,map_location"="cpu")
model = BertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
print("Model imported")


# tokenize text for training

inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=80, truncation=True,
                    padding='max_length')
print(inputs)
print(inputs.keys())

print("text tokenized")

inputs['labels'] = inputs['input_ids'].detach().clone()

"""We want to create our mask. Each token that is not a special token has a 15% chance of being masked. We don't want to mask [CLS], [SEP], or padding tokens. Correspond to the numbers 4, 5, and 1."""

rand = torch.rand(inputs['input_ids'].shape)

mask_arr = (rand < 0.15) * (inputs['input_ids'] != 4) * (inputs['input_ids'] != 5)* (inputs['input_ids'] != 1)

# masked tokens

selection = []

for i in range(mask_arr.shape[0]):
  selection.append(
    torch.flatten(mask_arr[i].nonzero()).tolist()
  )

for i in range(mask_arr.shape[0]):
  inputs['input_ids'][i,selection[i]] = 0

dataset = TweetsDataset(inputs)

print('dataset')

dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True)

print('dataloader')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('device')

model.to(device)

print('model to device')

model.train()

from torch.optim import AdamW

optim = AdamW(model.parameters(), lr=5e-5)

# puedo cambiar optimizador tambien

# Training loop

from tqdm import tqdm

epochs = 3

for epoch in range(epochs):
  loop = tqdm(dataloader, leave=True)
  for batch in loop:
    print('Batch')
    print(batch)
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    print('Input Ids')
    print(input_ids)
    print(input_ids.size()) 
    attention_mask = batch['attention_mask'].to(device)
    print(attention_mask)
    print('attention')
    labels= batch['labels'].to(device)
    print(labels)
    print('Labels')
    

    outputs = model(input_ids, attention_mask=attention_mask,
                   lm_labels=labels)
    print('Outputs')
    print(outputs)
    loss, prediction_scores = outputs[:2]
    print('Loss')
    loss.backward()
    optim.step()

    loop.set_description(f'Epoch {epoch}')
    loop.set_postfix(loss=loss.item())

save_directory = "pretrained_model"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)


file = open("finished.txt", "w")
file.write("Finished!")
file.close()



