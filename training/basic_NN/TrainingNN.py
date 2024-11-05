import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def robust_mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: torch.Tensor,
    labels_err: torch.Tensor,
    epsilon: float=1e-8,
) -> torch.Tensor:
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    total_var = torch.exp(variance) + torch.square(labels_err) + epsilon
    wrapper_output = 0.5 * (
        (torch.square(y_true - y_pred) / total_var) + torch.log(total_var)
    )

    losses = wrapper_output.sum() / y_true.shape[0]
    return losses

class SpectraDataset(Dataset):
    def __init__(self, spectra, labels, labels_err, spectra_mean=None, labels_mean=None, labels_std=None):
        if spectra_mean is None:
            spectra_mean = np.mean(spectra, axis=0)
        self.spectra_mean = spectra_mean
        self.spectra = (spectra - self.spectra_mean)  # Normalization

        if labels_mean is None or labels_std is None:
            labels_mean = np.nanmean(labels, axis=0)
            labels_std = np.nanstd(labels, axis=0)

        self.labels_mean = labels_mean
        self.labels_std = labels_std
        self.labels = (labels - self.labels_mean) / self.labels_std
        self.labels_err = labels_err / self.labels_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.spectra[idx], dtype=torch.float),
                torch.tensor(self.labels[idx], dtype=torch.float),
                torch.tensor(self.labels_err[idx], dtype=torch.float))

    def inverse_transform_labels(self, normalized_labels, normalized_labels_err):
        return normalized_labels * self.labels_std + self.labels_mean, normalized_labels_err * self.labels_std

    def save_means_stddevs(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, spectra_mean=self.spectra_mean, labels_mean=self.labels_mean, labels_std=self.labels_std)

torch.backends.cuda.matmul.allow_tf32 = False  # use tf32 for CUDA matrix multiplication
torch.backends.cudnn.allow_tf32 = False  # use tf32 for CUDNN

spectra_indices = np.arange(4107,4619)
pt_spectra_indices = spectra_indices
spectra_size = len(spectra_indices)

spectra_train = np.load("../../dataset/aspcap_spectra_train.npy")
#replace nan with 1.0
spectra_train[np.isnan(spectra_train)] = 1.0
labels_train = np.load("../../dataset/labels_train.npy")
labels_err_train = np.load("../../dataset/label_errs_train.npy")
wavelength_sol = np.loadtxt("../../dataset/apogee_wavelength_sol.csv", delimiter=",")
label_columns = [
    'TEFF', 'LOGG',
    'O_FE', 'MG_FE', 'SI_FE', 
    'TI_FE', 'TIII_FE', 
     'FE_H',  'NI_FE',
]

NI_index = label_columns.index('NI_FE')
SI_index = label_columns.index('SI_FE')
TI_index = label_columns.index('TI_FE')
TIII_index = label_columns.index('TIII_FE')
TEFF_index = label_columns.index('TEFF')
LOGG_index = label_columns.index('LOGG')
O_FE_index = label_columns.index('O_FE')
MG_FE_index = label_columns.index('MG_FE')

label_columns =  np.delete(label_columns, [SI_index, TI_index, TIII_index, NI_index, TEFF_index, LOGG_index, O_FE_index, MG_FE_index])
labels_train = np.delete(labels_train, [SI_index, TI_index, TIII_index, NI_index, TEFF_index, LOGG_index, O_FE_index, MG_FE_index], axis=1)
labels_err_train = np.delete(labels_err_train, [SI_index, TI_index, TIII_index, NI_index, TEFF_index, LOGG_index, O_FE_index, MG_FE_index], axis=1)
spectra_train = spectra_train[:, spectra_indices]

# Get the same 100 iron-rich stars used for training the Transformer model
dataset_size = 100
feh_training = labels_train[:, 0]
feh_training_idx = np.argwhere(feh_training > -1.0)
np.random.seed(77)
training_idxs = feh_training_idx[np.random.choice(np.arange(len(feh_training_idx)), 100, replace=False)]
spectra_train = spectra_train[training_idxs]
labels_train = labels_train[training_idxs]
labels_err_train = labels_err_train[training_idxs]
#(100,1,512) -> (100,512)
labels_train = np.squeeze(labels_train, axis=1)
labels_err_train = np.squeeze(labels_err_train, axis=1)
spectra_train = np.squeeze(spectra_train, axis=1)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class NormalNN(nn.Module):
    def __init__(self, input_size):
        super(NormalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.dropout1 = nn.Dropout(0.1)  # Add dropout layer
        self.fc2 = nn.Linear(2048, 1280)
        self.dropout2 = nn.Dropout(0.1)  # Add dropout layer
        self.fc3 = nn.Linear(1280, 1024)
        self.fc_out = nn.Linear(1024, 1)
        self.fc_logvar = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout
        x = F.relu(self.fc3(x))
        output = self.fc_out(x)
        logvar = self.fc_logvar(x)
        return output, logvar
    
# ================== hardware-related settings ==================
learning_rate = 1e-4  # initial learning rate
learning_rate_min = 1e-10  # minimum learning rate allowed
epochs = 1000  # number of epochs to train
cosine_annealing_t0 = 500  # cosine annealing restart length in epochs
checkpoint_every_n_epochs = 20  # save a checkpoint every n epochs
batch_size = 64  # number of samples per batch
device = 'cpu'
dataset = SpectraDataset(spectra_train, labels_train, labels_err_train)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset.save_means_stddevs("means_stddevs.npz")
#load 
means_stddevs = np.load("means_stddevs.npz")
spectra_mean = means_stddevs['spectra_mean']
labels_mean = means_stddevs['labels_mean']
labels_std = means_stddevs['labels_std']
spectra_test = np.load("../../dataset/aspcap_spectra_test.npy")
spectra_test[np.isnan(spectra_test)] = 1.0
labels_test = np.load("../../dataset/labels_test.npy")
labels_err_test = np.load("../../dataset/label_errs_test.npy")
labels_test = np.delete(labels_test, [SI_index, TI_index, TIII_index, NI_index, TEFF_index, LOGG_index, O_FE_index, MG_FE_index], axis=1)
labels_err_test = np.delete(labels_err_test, [SI_index, TI_index, TIII_index, NI_index, TEFF_index, LOGG_index, O_FE_index, MG_FE_index], axis=1)
spectra_test = spectra_test[:, spectra_indices]
test_dataset = SpectraDataset(spectra_test, labels_test, labels_err_test, spectra_mean, labels_mean, labels_std)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#val loader is 5% of the test loader
spectra_val = spectra_test[:len(spectra_test)//10]
labels_val = labels_test[:len(labels_test)//10]
labels_err_val = labels_err_test[:len(labels_err_test)//10]
val_dataset = SpectraDataset(spectra_val, labels_val, labels_err_val, spectra_mean, labels_mean, labels_std)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = NormalNN(input_size=512).to(device)
print("size val set", len(val_dataset))
#print number of params that need grad
numparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("numparams", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=cosine_annealing_t0,
    T_mult=1,
    eta_min=learning_rate_min,
    last_epoch=-1,
)

scheduler = lr_scheduler(optimizer)

# if median of last x:10 val losses is greater than median of last y:20 val losses, then overfit detected but continue training
last_x_val_losses = [float('inf')]*10
last_y_val_losses = [float('inf')]*20
import torch
import torch.nn.functional as F
import os
import csv
from datetime import datetime

# Create a directory for saving checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Create and open a CSV file to log losses
csv_file = open('training_metrics_NN.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['time', 'epoch', 'loss','val_loss','lr', 'overfit_detected'])

# Initialize lists to store loss values
train_losses = []
val_losses = []
min_val_loss = float('inf')
min_val_loss_epoch = 0
# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    overfit_detected = False

    
    for spectra, labels, labels_err in train_loader:
        spectra, labels, labels_err = spectra.to(device), labels.to(device), labels_err.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, logvars = model(spectra)
        non_nan_mask = ~torch.isnan(labels)
        y_true_non_nan = labels[non_nan_mask]
        y_pred_non_nan = outputs[non_nan_mask]
        y_vars_non_nan = logvars[non_nan_mask]
        y_errs_non_nan = labels_err[non_nan_mask]
        loss = robust_mean_squared_error(y_true_non_nan, y_pred_non_nan, y_vars_non_nan, y_errs_non_nan)
        
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    # Validation
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        for spectra, labels, labels_err in val_loader:
            spectra, labels, labels_err = spectra.to(device), labels.to(device), labels_err.to(device)
            outputs, logvars = model(spectra)
            non_nan_mask = ~torch.isnan(labels)
            y_true_non_nan = labels[non_nan_mask]
            y_pred_non_nan = outputs[non_nan_mask]
            y_vars_non_nan = logvars[non_nan_mask]
            y_errs_non_nan = labels_err[non_nan_mask]
            val_loss = robust_mean_squared_error(y_true_non_nan, y_pred_non_nan, y_vars_non_nan, y_errs_non_nan)
            running_val_loss += val_loss.item()
            
        # Update last x val losses
        last_x_val_losses.pop(0)
        last_x_val_losses.append(running_val_loss)
        last_y_val_losses.pop(0)
        last_y_val_losses.append(running_val_loss)
        
        # Check for early stopping
        if np.median(last_x_val_losses) > np.median(last_y_val_losses):
            overfit_detected = True

    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_val_loss = running_val_loss / len(val_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Append losses to the lists
    train_losses.append(epoch_loss)
    val_losses.append(epoch_val_loss)
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f} '
          f'Val Loss: {epoch_val_loss:.4f} '
          f'LR: {current_lr:.6f}, Overfitting: {overfit_detected}')
    if epoch_val_loss < min_val_loss:
        min_val_loss = epoch_val_loss
        min_val_loss_epoch = epoch
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print(f'Min val loss: {min_val_loss:.4f} at epoch {min_val_loss_epoch+1}')
    # Write losses and other metrics to the CSV file
    csv_writer.writerow([datetime.now(), epoch+1, round(epoch_loss, 3), round(epoch_val_loss, 3), 
                     round(current_lr, 3), overfit_detected])
    csv_file.flush()
    overfit_detected = False


# Save final model
torch.save(model.state_dict(), 'checkpoints/final_model.pth')

# Close the CSV file
csv_file.close()

print("Training complete")

#load best model
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Initialize lists to hold predictions and true values for each label
num_labels = 1  # Just Fe
y_preds = [[] for _ in range(num_labels)]
y_trues = [[] for _ in range(num_labels)]
y_true_errs = [[] for _ in range(num_labels)]
y_pred_errs = [[] for _ in range(num_labels)]

with torch.no_grad():
    for j, (inputs, targets, targets_err) in enumerate(test_loader):
        outputs, logvars = model(inputs)
        outputs_transformed, outputs_err_transformed = test_dataset.inverse_transform_labels(outputs, torch.sqrt(torch.exp(logvars)))
        targets_transformed, targets_err_transformed = test_dataset.inverse_transform_labels(targets, targets_err)
        
        # Assuming outputs and targets are 2D tensors: [batch_size, num_labels]
        for i in range(num_labels):
            y_preds[i].extend(outputs_transformed[:, i].flatten().tolist())
            y_trues[i].extend(targets_transformed[:, i].flatten().tolist())
            y_true_errs[i].extend(targets_err_transformed[:, i].flatten().tolist())
            y_pred_errs[i].extend(outputs_err_transformed[:, i].flatten().tolist())

import pandas as pd
y_trues = np.array(y_trues).T
y_preds = np.array(y_preds).T
y_true_errs = np.array(y_true_errs).T
y_pred_errs = np.array(y_pred_errs).T
#make df with columns
# FE_H_pred,FE_H_err,FE_H_actual,FE_H_actual_err
df = pd.DataFrame(data=np.concatenate((y_preds, y_pred_errs, y_trues, y_true_errs), axis=1), columns=["FE_H_pred","FE_H_err","FE_H_actual","FE_H_actual_err"])
numparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
df.to_csv(f"NN_pred_true_may28_{numparams}.csv", index=False)

