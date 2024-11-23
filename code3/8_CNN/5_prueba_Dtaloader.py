"""
5_prueba_Dtaloader.py

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from SanaM import SpectralDataset
from torch.utils.data import DataLoader

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

dataset = SpectralDataset(inputs=X_train, labels=y_train)

data_loarder = DataLoader(dataset, batch_size=8, shuffle=True)

data_inputs, data_labels = next(iter(data_loarder))


print("Data inputs", data_inputs.shape, "\n", data_inputs)
print()
print("Data labels", data_labels.shape, "\n", data_labels)
