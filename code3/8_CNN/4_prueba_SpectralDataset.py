"""
4_prueba_SpectralDataset.py

"""

import os
import pandas as pd
from SanaM import SpectralDataset, SanaModel

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

X_df = df.iloc[:9, 4:].values
y_df = df.iloc[:9, 1].values

dataset = SpectralDataset(inputs=X_df, labels=y_df)


# AAAAAAAAAAAAAAAAAAAAAAAAAAA

