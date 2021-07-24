import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as ply
import numpy as np

data = [("What the fuck".lower().split() , ["O","O","CS"]),
        ("The boy asked him to fuckoff".lower().split() ,["O","O","O","O","O","CS"]),
        ("I hate that bastard".lower().split() , ["O","O","O","CS"]),
        ("He is a dicked".lower().split(),["O","O","O","CS"]),
        ("Hey prick".lower().split(),["O","CS"]),
        ("What a pussy you are".lower().split() , ["O","O","CS","O","O"]),
        ("Dont be a cock".lower().split(),["O","O","O","CS"])]

word2idx = {}

for sent , tag in data:
  for word in sent:
    if word not in word2idx:
      word2idx[word] = len(word2idx)

tag2idx = {"O" : 0 , "CS" : 1}
tag2rev = {0 : "O" , 1 : "CS"}

def prepare_sequence(seq , to_idx):
  idxs = [to_idx[word] for word in seq]
  idxs = np.array(idxs)
  return torch.tensor(idxs).long()

testsent = "fuckoff boy".lower().split()
inp = prepare_sequence(testsent , word2idx)
print("The test sentence {} is tranlated to {}\r\n".format(testsent , inp))

class LSTMTagger(nn.Module):

  def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):

    super(LSTMTagger , self).__init__()

    self.hidden_dim = hidden_dim

    self.word_embedding = nn.Embedding(vocab_size , embedding_dim= embedding_dim)

    self.lstm = nn.LSTM(input_size= embedding_dim , hidden_size = hidden_dim)

    self.hidden2tag = nn.Linear(hidden_dim , tagset_size)

    self.hidden = self.init_hidden()

  def init_hidden(self):

    return (torch.randn(1 , 1 , self.hidden_dim),
           torch.randn(1 , 1 , self.hidden_dim))

  def forward(self , sentence):

    embeds = self.word_embedding(sentence)

    lstm_out , hidden_out = self.lstm(embeds.view(len(sentence) , 1 , -1) , self.hidden) 

    tag_outputs = self.hidden2tag(lstm_out.view(len(sentence) , -1))
    tag_scores = F.log_softmax(tag_outputs , dim = 1)

    return tag_scores   

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
model = LSTMTagger(EMBEDDING_DIM , HIDDEN_DIM , len(word2idx) , len(tag2idx))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters() , lr = 0.1)

n_epochs = 500

for epoch in range(n_epochs):

  epoch_loss = 0.0

  for sent , tags in data:

    model.zero_grad()
    input_sent = prepare_sequence(sent , word2idx)
    tag = prepare_sequence(tags , tag2idx)

    model.hidden = model.init_hidden()

    output = model(input_sent)

    loss = loss_function(output , tag)

    epoch_loss += loss.item()

    loss.backward()

    optimizer.step()

  if epoch % 20 == 19:
    print("Epoch : {} , loss : {}".format(epoch , epoch_loss / len(data)))

testsent = "cock".lower().split()
inp = prepare_sequence(testsent , word2idx)

print("Input sent : {}".format(testsent))
tags = model(inp)
_,pred_tags = torch.max(tags , 1)
print("Pred tag : {}".format(pred_tags))
pred = np.array(pred_tags)

for i in range(len(testsent)):
  print("Word : {} , Predicted tag : {}".format(testsent[i] , tag2rev[pred[i]]))