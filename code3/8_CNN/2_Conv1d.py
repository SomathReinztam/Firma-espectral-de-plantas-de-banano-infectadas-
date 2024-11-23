"""
2_Conv1d.py  


"""

import os
import pandas as pd
from torch.nn import Conv1d, MaxPool1d
import torch

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

conv1 = Conv1d(
    in_channels=1,
    out_channels=2,
    kernel_size=9,
    stride=1,
    padding=4
)

Mxp = MaxPool1d(
    kernel_size=2
)

df = pd.read_csv(file, sep=';')
#lgtd_ond = [float(i) for i in df.columns[4:].values]

filtro = (df['Tratamiento'] == 2) & (df['Planta'] == 19) & (df['dpi'] == 7)

df = df[filtro].iloc[:, 4:].values[0]

df = torch.from_numpy(df).view(1, 1, -1).float()

print(df.shape)
print()

#df = conv1(df)
df = Mxp(df)

print(df.shape)
print()
