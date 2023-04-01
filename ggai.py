
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import pickle
from torch.utils.data import DataLoader, TensorDataset

#Data Load and Processing

rawX = pd.read_csv('gg/ggX_lite.csv')
rawy = pd.read_csv('gg/ggY_lite.csv')
X = np.asarray(rawX).astype(np.float32)
y = np.asarray(rawy).astype(np.float32)
Xscaler = StandardScaler()
X = Xscaler.fit_transform(X)
Yscaler = StandardScaler()
y = Yscaler.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
with open('gg/Xscaler_lite.pkl', 'wb') as f:
    pickle.dump(Xscaler, f)

with open('gg/Yscaler_lite.pkl', 'wb') as f:
    pickle.dump(Yscaler, f)

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
input_dim = X_train.shape[1]
outcome_dim = y_train.shape[1]
hidden_dim = 1024
latent_dim = 128
epochs_pretrain = 200
epochs_finetune = 1800
lr_pretrain = 0.0001
lr_finetune = 0.00001
dropout_prob_pretrain = 0.0
dropout_prob_finetune = 0.1
l2_reg_pretrain = 0.0
l2_reg_finetune = 0.0001

lls = []

#Pretraining

ae_pretrain = Autoencoder(input_dim, hidden_dim, latent_dim, outcome_dim, dropout_prob_pretrain, l2_reg_pretrain)
optimizer_pretrain = optim.Adam(ae_pretrain.parameters(), lr=lr_pretrain)
criterion_pretrain = nn.MSELoss()

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(epochs_pretrain):
    for x, y_true in train_loader:
        x = x.float()
        y_true = y_true.float()
        recon_x, y_pred, l2_loss = ae_pretrain(x)
        loss1 = criterion_pretrain(recon_x, x)
        loss2 = criterion_pretrain(y_pred, y_true)
        loss = criterion_pretrain(recon_x, x) + criterion_pretrain(y_pred, y_true) + l2_loss
        optimizer_pretrain.zero_grad()
        loss.backward()
        optimizer_pretrain.step()
    valloss = np.mean(abs(ae_pretrain.y_network(ae_pretrain.encoder(torch.from_numpy(X_test))).detach().numpy() - y_test)) / np.mean(abs(np.mean(y_test) - y_test))
    trainloss = np.mean(abs(ae_pretrain.y_network(ae_pretrain.encoder(torch.from_numpy(X_train))).detach().numpy() - y_train)) / np.mean(abs(np.mean(y_train) - y_train))
    lls.append([valloss,trainloss])
    print('Pretraining @ Epoch [{}/{}], Loss: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}] @ {}'.format(epoch + 1, epochs_pretrain, loss.item(), loss1.item(), loss2.item(), l2_loss.item(), valloss, trainloss, datetime.datetime.now()))

#Fine tuning

ae_finetune = Autoencoder(input_dim, hidden_dim, latent_dim, outcome_dim, dropout_prob_finetune, l2_reg_finetune)
ae_finetune.encoder.load_state_dict(ae_pretrain.encoder.state_dict())
ae_finetune.y_network.load_state_dict(ae_pretrain.y_network.state_dict())
ae_finetune.decoder.load_state_dict(ae_pretrain.decoder.state_dict())

optimizer_finetune = optim.Adam(ae_finetune.parameters(), lr=lr_finetune)
criterion_finetune = nn.MSELoss()
train_dataset_finetune = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader_finetune = DataLoader(train_dataset_finetune, batch_size=32, shuffle=True)

for epoch in range(epochs_finetune):
    for x, y_true in train_loader_finetune:
        x = x.float()
        y_true = y_true.float()
        recon_x, y_pred, l2_loss = ae_finetune(x)
        loss1 = criterion_finetune(recon_x, x)
        loss2 = criterion_finetune(y_pred, y_true)
        loss = criterion_finetune(recon_x, x) + 5 * criterion_finetune(y_pred, y_true) + l2_loss
        optimizer_finetune.zero_grad()
        loss.backward()
        optimizer_finetune.step()
    valloss = np.mean(abs(ae_finetune.y_network(ae_finetune.encoder(torch.from_numpy(X_test))).detach().numpy() - y_test)) / np.mean(abs(np.mean(y_test) - y_test))
    trainloss = np.mean(abs(ae_finetune.y_network(ae_finetune.encoder(torch.from_numpy(X_train))).detach().numpy() - y_train)) / np.mean(abs(np.mean(y_train) - y_train))
    lls.append([valloss,trainloss])
    print('Fine-tuning @ Epoch [{}/{}], Loss: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}] @ {}'.format(epochs_pretrain+epoch + 1, epochs_pretrain+epochs_finetune, loss.item(), loss1.item(), loss2.item(), l2_loss.item(), valloss, trainloss, datetime.datetime.now()))

torch.save(ae_pretrain.state_dict(), 'ae_pretrain_lite.pt')
torch.save(ae_finetune.state_dict(), 'ae_finetune_lite.pt')

#Validation

ae = ae_pretrain
print(f"{np.mean(abs(ae.y_network(ae.encoder(torch.from_numpy(X_test))).detach().numpy() - y_test))/np.mean(abs(np.mean(y_test) - y_test))},{np.mean(abs(ae.y_network(ae.encoder(torch.from_numpy(X_train))).detach().numpy() - y_train))/np.mean(abs(np.mean(y_train) - y_train))}")
ae = ae_finetune
print(f"{np.mean(abs(ae.y_network(ae.encoder(torch.from_numpy(X_test))).detach().numpy() - y_test))/np.mean(abs(np.mean(y_test) - y_test))},{np.mean(abs(ae.y_network(ae.encoder(torch.from_numpy(X_train))).detach().numpy() - y_train))/np.mean(abs(np.mean(y_train) - y_train))}")
