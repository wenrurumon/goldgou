
import torch
import pandas as pd
import numpy as np
import sys
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler
import datetime
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import rankdata
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import logging
import datetime
from collections import Counter
import math

###################################
#Module

class process():
    def __init__(self,filename):
        raw = pd.read_csv(filename)
        X = []
        for i in raw.columns.tolist():
            if 'f' in i:
                X.append(np.ravel(raw[i]))
        X = np.asarray(X).T
        Xscaler = StandardScaler()
        Xscaler.fit(X)
        X = Xscaler.transform(X)
        Y = np.ravel(raw['is_gain'])
        self.X = X
        self.Y = Y
        self.Xscaler = Xscaler

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, l2_penalty):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_penalty = l2_penalty
    def forward(self, x):
        x = self.dropout(x)
        hidden_output = torch.relu(self.linear1(x))
        output = torch.sigmoid(self.linear2(hidden_output))
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return output - self.l2_penalty * l2_reg

###################################
#Parameter

filename = 'data/data_train_demo.csv'
seedi = 777
device = torch.device('cpu')
hidden_dim = 1
dropout_rate = 0.0
l2_penalty = 0.001
lr = 0.01

dataset = process(filename)
X = torch.tensor(dataset.X).float().to(device)
Y = torch.tensor(dataset.Y).float().to(device).view(-1, 1)
input_dim = X.shape[1]
hidden_dim = 8

###################################
#Data Process

# X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=seedi)
X_train,X_test,Y_train,Y_test = X[range(800),:],X[800:,:],Y[range(800),:],Y[800:,:]
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)

###################################
#Benchmark

sklmodel = linear_model.LogisticRegression()
sklmodel.fit(X_train,np.ravel(Y_train))

###################################
#Trianing

model = LogisticRegressionModel(input_dim,hidden_dim,dropout_rate,l2_penalty)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model.pth')
loaded_model = LogisticRegressionModel(input_dim,hidden_dim,dropout_rate,l2_penalty)
loaded_model.load_state_dict(torch.load('model.pth'))


###################################
#Validate

def precision(y_pred, y_true, thres=0.5):
    y_pred = np.ravel(y_pred) > thres
    y_true = np.ravel(y_true)
    return(np.sum((y_pred==y_true)&(y_pred))/np.sum(np.ravel(y_true)))

precision(sklmodel.predict(X_train),Y_train,0.4)
precision(model(X_train).detach().numpy(),Y_train,0.4)
precision(sklmodel.predict(X_test),Y_test,0.4)
precision(model(X_test).detach().numpy(),Y_test,0.4)
