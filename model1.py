
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

#HyperParameter

hidden_dim = 2048
latent_dim = 128
dropout_rates = [0.1,0.2,0.3,0.4]
lrs = [0.001,0.0005,0.0001,0.0]
l2_regs = [0.0001,0.0005,0.001,0.001]
num_epochss = [2000,3000,3000,3000]
early_stops = [1000,1000,1000,1000]
batch_sizes = [32,32,32,32]
patiences = [50,40,30,20]

#Data Load and Processing

arg1 = sys.argv[1]
# arg1 = '0406'
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

# Tensoring

X = torch.tensor(X).float().cuda()
Y = torch.tensor(Y).float().cuda()
X_val = X[-100:,:]
Y_val = Y[-100:,:]

# Model Setup

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, dropout_rate, l2_reg):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(p=dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + output_dim, hidden_dim), # 加上output_dim作为输入
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(p=dropout_rate)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=dropout_rate)
        )
        self.regularization = nn.ModuleList()
        self.regularization.append(nn.Linear(hidden_dim, hidden_dim))
        self.regularization.append(nn.Linear(latent_dim, latent_dim))
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.predictor(z)
        concat = torch.cat((z, y_pred), dim=1)
        x_pred = self.decoder(concat)
        reg_loss = 0.0
        for layer in self.regularization:
            reg_loss += torch.norm(layer.weight)
        reg_loss *= self.l2_reg
        return x_pred, y_pred, reg_loss

# Resulting

models = []
input_dim = X.shape[1]
output_dim = Y.shape[1]
print('dropout_rates',dropout_rates)
print('learning_rates',lrs)
print('L2 Penalties',l2_regs)

# Pretrain Model

k = 0
dropout_rate = dropout_rates[k]
lr = lrs[k]
l2_reg = l2_regs[k]
model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

print(f"Pretrain Model Start at: {datetime.datetime.now()}")
num_epochs = num_epochss[k]
batch_size = batch_sizes[k]
patience = patiences[k]
best_loss = np.inf
counter = 0

for epoch in range(num_epochs):
    # Random shuffle the data
    indices = torch.randperm(X.shape[0])
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    # Iterate over batches
    for i in range(0, X.shape[0], batch_size):
        # Get the current batch
        X_batch = X_shuffled[i:i+batch_size].cuda()
        Y_batch = Y_shuffled[i:i+batch_size].cuda()
        x_hat, y_pred, reg_loss = model(X_batch)
        loss_x = criterion(x_hat, X_batch) 
        loss_y = criterion(y_pred, Y_batch) 
        loss = 8*loss_x + 2*loss_y + reg_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    with torch.no_grad():
        x_val_hat, y_val_pred, _ = model(X_val.cuda())
        val_loss_x = criterion(x_val_hat, X_val.cuda())
        val_loss_y = criterion(y_val_pred, Y_val.cuda())
        val_loss = 8*val_loss_x + 2*val_loss_y
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter > patience and epoch > early_stops[k]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}/{loss_x.item():.4f}/{loss_y.item():.4f}, Validation Loss: {val_loss.item():.4f}/{val_loss_x.item():.4f}/{val_loss_y.item():.4f}/{best_loss.item():.4f} Time: {datetime.datetime.now()}")

print(f"Pretrain Model End at: {datetime.datetime.now()}")
models.append(best_model_state_dict)
torch.save(best_model_state_dict, f'model/cuda_model1_{k}_{arg1}.pt')
print(f'Saved: model/cuda_model1_{k}_{arg1}.pt')

# FineTune 1

k = 1
dropout_rate = dropout_rates[k]
lr = lrs[k]
l2_reg = l2_regs[k]
model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).cuda()
model.load_state_dict(best_model_state_dict)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

print(f"Finetune Model 1 Start at: {datetime.datetime.now()}")
num_epochs = num_epochss[k]
batch_size = batch_sizes[k]
patience = patiences[k]
best_loss = np.inf
counter = 0

for epoch in range(num_epochs):
    # Random shuffle the data
    indices = torch.randperm(X.shape[0])
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    # Iterate over batches
    for i in range(0, X.shape[0], batch_size):
        # Get the current batch
        X_batch = X_shuffled[i:i+batch_size].cuda()
        Y_batch = Y_shuffled[i:i+batch_size].cuda()
        x_hat, y_pred, reg_loss = model(X_batch)
        loss_x = criterion(x_hat, X_batch) 
        loss_y = criterion(y_pred, Y_batch) 
        loss = 5*loss_x + 5*loss_y + reg_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    with torch.no_grad():
        x_val_hat, y_val_pred, _ = model(X_val.cuda())
        val_loss_x = criterion(x_val_hat, X_val.cuda())
        val_loss_y = criterion(y_val_pred, Y_val.cuda())
        val_loss = 5*val_loss_x + 5*val_loss_y
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter > patience and epoch > early_stops[k]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}/{loss_x.item():.4f}/{loss_y.item():.4f}, Validation Loss: {val_loss.item():.4f}/{val_loss_x.item():.4f}/{val_loss_y.item():.4f}/{best_loss.item():.4f} Time: {datetime.datetime.now()}")

print(f"Finetune Model 1 End at: {datetime.datetime.now()}")
models.append(best_model_state_dict)
torch.save(best_model_state_dict, f'model/cuda_model1_{k}_{arg1}.pt')
print(f'Saved: model/cuda_model1_{k}_{arg1}.pt')

# FineTune 2

k = 2
dropout_rate = dropout_rates[k]
lr = lrs[k]
l2_reg = l2_regs[k]
model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).cuda()
model.load_state_dict(best_model_state_dict)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

print(f"Finetune Model 2 Start at: {datetime.datetime.now()}")
num_epochs = num_epochss[k]
batch_size = batch_sizes[k]
patience = patiences[k]
best_loss = np.inf
counter = 0

for epoch in range(num_epochs):
    # Random shuffle the data
    indices = torch.randperm(X.shape[0])
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    # Iterate over batches
    for i in range(0, X.shape[0], batch_size):
        # Get the current batch
        X_batch = X_shuffled[i:i+batch_size].cuda()
        Y_batch = Y_shuffled[i:i+batch_size].cuda()
        x_hat, y_pred, reg_loss = model(X_batch)
        loss_x = criterion(x_hat, X_batch) 
        loss_y = criterion(y_pred, Y_batch) 
        loss = 2*loss_x + 8*loss_y + reg_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    with torch.no_grad():
        x_val_hat, y_val_pred, _ = model(X_val.cuda())
        val_loss_x = criterion(x_val_hat, X_val.cuda())
        val_loss_y = criterion(y_val_pred, Y_val.cuda())
        val_loss = 2*val_loss_x + 8*val_loss_y
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter > patience and epoch > early_stops[k]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}/{loss_x.item():.4f}/{loss_y.item():.4f}, Validation Loss: {val_loss.item():.4f}/{val_loss_x.item():.4f}/{val_loss_y.item():.4f}/{best_loss.item():.4f} Time: {datetime.datetime.now()}")

print(f"Finetune Model 2 End at: {datetime.datetime.now()}")
models.append(best_model_state_dict)
torch.save(best_model_state_dict, f'model/cuda_model1_{k}_{arg1}.pt')
print(f'Saved: model/cuda_model1_{k}_{arg1}.pt')

# FineTune 3

k = 3
dropout_rate = dropout_rates[k]
lr = lrs[k]
l2_reg = l2_regs[k]
model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).cuda()
model.load_state_dict(best_model_state_dict)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

print(f"Finetune Model 3 Start at: {datetime.datetime.now()}")
num_epochs = num_epochss[k]
batch_size = batch_sizes[k]
patience = patiences[k]
best_loss = np.inf
counter = 0

X2 = X[-200:,:]
Y2 = Y[-200:,:]

for epoch in range(num_epochs):
    # Random shuffle the data
    indices = torch.randperm(X2.shape[0])
    X_shuffled = X2[indices]
    Y_shuffled = Y2[indices]
    # Iterate over batches
    for i in range(0, X2.shape[0], batch_size):
        # Get the current batch
        X_batch = X_shuffled[i:i+batch_size].cuda()
        Y_batch = Y_shuffled[i:i+batch_size].cuda()
        x_hat, y_pred, reg_loss = model(X_batch)
        loss_x = criterion(x_hat, X_batch) 
        loss_y = criterion(y_pred, Y_batch) 
        loss = 2*loss_x + 8*loss_y + reg_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    with torch.no_grad():
        x_val_hat, y_val_pred, _ = model(X_val.cuda())
        val_loss_x = criterion(x_val_hat, X_val.cuda())
        val_loss_y = criterion(y_val_pred, Y_val.cuda())
        val_loss = 2*val_loss_x + 8*val_loss_y
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter > patience and epoch > early_stops[k]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}/{loss_x.item():.4f}/{loss_y.item():.4f}, Validation Loss: {val_loss.item():.4f}/{val_loss_x.item():.4f}/{val_loss_y.item():.4f}/{best_loss.item():.4f} Time: {datetime.datetime.now()}")

print(f"Finetune Model 3 End at: {datetime.datetime.now()}")
models.append(best_model_state_dict)
torch.save(best_model_state_dict, f'model/cuda_model1_{k}_{arg1}.pt')
print(f'Saved: model/cuda_model1_{k}_{arg1}.pt')
np.savez(f'model/cuda_para1_{k}_{arg1}.npz',dropout_rates=dropout_rates,lrs=lrs,l2_regs=l2_regs)

#Trade Strategy

Z = torch.tensor(Z).float()
Ypreds = []
for modeli in models:
    Ypreds.append(Yscaler.inverse_transform(model(Z.cuda())[1].cpu().detach().numpy()))

for i in range(len(Ypreds)):
    print(f"Ypreds[{i}]:")
    top_cols = (-Ypreds[i]).argsort(axis=1)[:, :10]
    top_vals = Ypreds[i][np.arange(Ypreds[i].shape[0])[:, None], top_cols]
    top_cols = codes[top_cols]
    print(top_cols)
    print(np.round(top_vals,4))

