"""
6_prueba_Model.py

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from SanaM import SpectralDataset, SanaModel, ModelTrainer
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import torch


path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

X_df = df.iloc[:, 4:].values
y_df = df['Sana'].values


X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.2,
    random_state=29
)

dataset_train = SpectralDataset(inputs=X_train, labels=y_train)
dataset_valid = SpectralDataset(inputs=X_test, labels=y_test)


train_dl = DataLoader(dataset_train, batch_size=32, shuffle=True)
valid_dl = DataLoader(dataset_valid, batch_size=32, shuffle=True)


model = SanaModel()

loss_fn = BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = ModelTrainer(model, loss_fn, optimizer)

num_epochs = 250


trainer.train_model(train_dl, valid_dl, num_epochs)
trainer.plot_learning_curve()


