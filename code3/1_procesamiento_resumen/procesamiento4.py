"""
procesamiento4.py

"""

import os
import pandas as pd

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

filtro = (df['Tratamiento'] != 0)

df = df[filtro]

new_file = 'datos_no_control.csv'
new_file = os.path.join(path, new_file)
df.to_csv(new_file, sep=';', index=False)

