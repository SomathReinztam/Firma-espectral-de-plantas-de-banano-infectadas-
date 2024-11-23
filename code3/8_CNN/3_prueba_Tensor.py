"""
3_prueba_Tensor.py  


"""

import os
import pandas as pd
import torch
from SanaM import SanaModel


path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')


filtro = (df['Tratamiento'] == 2) & (df['Planta'] == 19) & (df['dpi'] == 7)

df = df[filtro].iloc[:, 4:].values[0]

df = torch.from_numpy(df).view(1, 1, -1).float()

#df = torch.from_numpy(df).float()


"""
torch.manual_seed(1)

model = SanaModel()

pred = model(df)[:, 0]

print(pred.shape)
print()
print(pred)
"""


