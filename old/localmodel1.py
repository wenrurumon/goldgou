
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

#Data Load and Processing

# arg1 = sys.argv[1]
arg1 = '0404'
raw = np.load(f'data/model1_{arg1}.npz',allow_pickle=True)
print(f'load data: data/model1_{arg1}.npz')
X = raw['X'][:,1:]
Y = raw['Y'][:,1:]
Z = raw['Z'][:,1:]
codes = raw['codes']
dates = raw['dates']

Xscaler = StandardScaler()
Xscaler.fit(X)
X = Xscaler.transform(X)
Z = Xscaler.transform(Z)
Yscaler = StandardScaler()
Yscaler.fit(Y)
Y = Yscaler.transform(Y)

models = []

#Network Structure

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, outcome_dim, dropout_prob, l2_reg):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout_prob)
        )
        self.y_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outcome_dim),
            nn.Dropout(dropout_prob)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+outcome_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_prob)
        )
        self.l2_reg = l2_reg
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z, y):
        combined = torch.cat((z, y), dim=1)
        return self.decoder(combined)
    def forward(self, x):
        z = self.encode(x)
        y_pred = self.y_network(z)
        recon_x = self.decode(z, y_pred)
        l2_loss = torch.tensor(0.)
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return recon_x, y_pred, self.l2_reg*l2_loss

# Hyper Parameters
input_dim = X.shape[1]
outcome_dim = Y.shape[1]
hidden_dim = 2048
latent_dim = 128
epochs = [10,10,10]
lr = [0.0001,0.00001]
dropout_prob = [.0,.2]
l2 = [0,0.00001]
workers = 4

#Pretraining

i = 0
modeli = Autoencoder(input_dim, hidden_dim, latent_dim, outcome_dim, dropout_prob[i], l2[i])
optimizeri = optim.Adam(modeli.parameters(), lr=lr[i])
criterioni = nn.MSELoss()
train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)

for epoch in range(epochs[i]):
    for x, y_true in train_loader:
        x = x.float()
        y_true = y_true.float()
        recon_x, y_pred, l2_loss = modeli(x)
        loss1 = criterioni(recon_x, x)
        loss2 = criterioni(y_pred, y_true)
        loss = 5 * criterioni(recon_x, x) + criterioni(y_pred, y_true) + l2_loss
        optimizeri.zero_grad()
        loss.backward()
        optimizeri.step()
    print('Pretraining @ Epoch [{}/{}], Loss: [{:.4f}/{:.4f}/{:.4f}/{:.4f}] @ {}'.format(epoch + 1, np.sum(epochs[:(i+1)]), loss.item(), loss1.item(), loss2.item(), l2_loss.item(), datetime.datetime.now()))

torch.save(modeli.state_dict(), f'model/model1_{i}_{arg1}.pt')
models.append(modeli)

#Finetuning

i = 1
modeli = Autoencoder(input_dim, hidden_dim, latent_dim, outcome_dim, dropout_prob[i], l2[i])
modeli.encoder.load_state_dict(models[i-1].encoder.state_dict())
modeli.y_network.load_state_dict(models[i-1].y_network.state_dict())
modeli.decoder.load_state_dict(models[i-1].decoder.state_dict())

optimizeri = optim.Adam(modeli.parameters(), lr=lr[i])
criterioni = nn.MSELoss()
train_dataset = TensorDataset(torch.from_numpy(X[-200:,:]), torch.from_numpy(Y[-200:,:]))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)

for epoch in range(epochs[i]):
    for x, y_true in train_loader:
        x = x.float()
        y_true = y_true.float()
        recon_x, y_pred, l2_loss = modeli(x)
        loss1 = criterioni(recon_x, x)
        loss2 = criterioni(y_pred, y_true)
        loss = criterioni(recon_x, x) + 5*criterioni(y_pred, y_true) + l2_loss
        optimizeri.zero_grad()
        loss.backward()
        optimizeri.step()
    print('Finetuning 1 @ Epoch [{}/{}], Loss: [{:.4f}/{:.4f}/{:.4f}/{:.4f}] @ {}'.format(np.sum(epochs[:i]) + epoch + 1, np.sum(epochs[:(i+1)]), loss.item(), loss1.item(), loss2.item(), l2_loss.item(), datetime.datetime.now()))

torch.save(modeli.state_dict(), f'model/model1_{i}_{arg1}.pt')
models.append(modeli)

#Validation and Prediction

Z = torch.from_numpy(Z).to(dtype=torch.float32)
Ypreds = []
for modeli in models:
    Ypreds.append(Yscaler.inverse_transform(modeli.y_network(modeli.encoder(Z)).detach().numpy()))

for i in range(len(Ypreds)):
    print(f"Ypreds[{i}]:")
    top_cols = (-Ypreds[i]).argsort(axis=1)[:, :10]
    top_vals = Ypreds[i][np.arange(Ypreds[i].shape[0])[:, None], top_cols]
    top_cols = codes[top_cols]
    print(top_cols)
    print(np.round(top_vals,4))



