"""
procesamiento3.py

El objetivo de este script es crear una nueva columna 
en datos.csv llamda sana en la cual 1 significa sana
y 0 que no está sana
"""

import os
import pandas as pd

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_path = os.path.join(path, file)

df = pd.read_csv(full_path, sep=';')

n = df.shape[0]
lista = []
for i in range(n):
    if (df.iloc[i, 0] == 0) or (df.iloc[i, 1] == 0):
        lista.append(1)
    else:
        lista.append(0)

df2 = pd.DataFrame(lista, columns=['Sana'])

df1 = df['dpi']
df3 = df.iloc[:,1:]


df_new = pd.concat([df1, df2, df3], axis=1)

df_new.to_csv(full_path, sep=';', index=False)
