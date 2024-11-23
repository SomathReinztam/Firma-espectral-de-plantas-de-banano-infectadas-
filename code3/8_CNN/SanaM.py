"""
SanaM.py

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Crear clase personalizada de Dataset
class SpectralDataset(Dataset):
    def __init__(self, inputs, labels):
        # Convertir los datos en tensores de PyTorch
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Ajustar el input a formato [num_channels, sequence_length] para CNN 1D
        x = self.inputs[idx].unsqueeze(0)  # agregar una dimensión de canal     .unsqueeze(0)   .view(1, 1, -1)
        y = self.labels[idx]
        return x, y



class SanaModel(nn.Module):
    def __init__(self):
        super(SanaModel, self).__init__()
        """
        Bloque convulucional

        """
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=20, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.3)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=24, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=57)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=24, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x



class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_hist_train = []
        self.accuracy_hist_train = []
        self.loss_hist_valid = []
        self.accuracy_hist_valid = []
    
    def train_model(self, train_dl, valid_dl, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses_train = []
            epoch_accuracies_train = []

            for data_inputs, data_labels in train_dl:
                pred = self.model(data_inputs)[:, 0]
                loss = self.loss_fn(pred, data_labels.float())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_losses_train.append(loss.item())
                is_correct = ((pred>=0.5).float() == data_labels).float()
                epoch_accuracies_train.append(is_correct.mean().item())
                data_inputs, data_labels, pred = None, None, None
            
            self.loss_hist_train.append(np.mean(epoch_losses_train))
            self.accuracy_hist_train.append(np.mean(epoch_accuracies_train))

            valid_loss, valid_accuracy = self.validate(valid_dl)
            self.loss_hist_valid.append(valid_loss)
            self.accuracy_hist_valid.append(valid_accuracy)
            print(f'Epoch {epoch+1} / {num_epochs}: Training Loss: {np.mean(epoch_losses_train):.4f}, Training Accuracy: {np.mean(epoch_accuracies_train):.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

        #return self.loss_hist_train, self.accuracy_hist_train, self.loss_hist_valid, self.accuracy_hist_valid
    
    def validate(self, valid_dl):
        self.model.eval()
        epoch_losses_valid = []
        epoch_accuracies_valid = []

        with torch.no_grad():
            for data_inputs, data_labels in valid_dl:
                pred = self.model(data_inputs)[:, 0]
                loss = self.loss_fn(pred, data_labels.float())
                epoch_losses_valid.append(loss.item())
                is_correct = ((pred>=0.5).float() == data_labels).float()
                epoch_accuracies_valid.append(is_correct.mean().item())
        return np.mean(epoch_losses_valid), np.mean(epoch_accuracies_valid)

    
    def plot_learning_curve(self):
        plt.style.use('seaborn-whitegrid')
        fig, (ax0, ax1) = plt.subplots(ncols=2,  figsize=(12, 10))
        ax0.plot(self.loss_hist_train, linestyle='solid', label='Training')
        ax0.plot(self.loss_hist_valid, linestyle='solid', label='Test')
        ax0.set_title("Loss history")
        ax0.legend()

        ax1.plot(self.accuracy_hist_train, linestyle='solid', label='Training')
        ax1.plot(self.accuracy_hist_valid, linestyle='solid', label='Test')
        ax1.set_title("accuracy history")
        ax1.set_title("Accuracy history")
        ax1.legend()

        plt.show()






# https://chatgpt.com/share/672e6067-71f0-8013-9ed8-4b1c006ac54e
