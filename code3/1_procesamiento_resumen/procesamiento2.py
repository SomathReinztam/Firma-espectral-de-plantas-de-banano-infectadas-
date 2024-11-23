"""
procesamiento2.py

El objetivo de este script es trasformar tolos los datos
de datos.csv de las longitudes de onada, de str a float
"""

import os
import pandas as pd

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_path = os.path.join(path, file)

file2 = "datos_describe.csv"
full_path2 = os.path.join(path, file2)


df = pd.read_csv(full_path, sep=';')

def puntos_por_comas(string):
    cadena = ''
    n = len(string)
    for i in range(n):
        if string[i] == ',':
            cadena = cadena + '.'
        else:
            cadena = cadena + string[i]
    return float(cadena)

def editar_lgtd_onda_columnas():
    col = df.columns.values
    n = len(col)
    for i in range(3, n):
        x = col[i]
        col[i] = puntos_por_comas(x)
    df.columns = col

def editar_lgtd_onda_valores():
    filas = df.shape[0]
    col = df.shape[1]
    for j in range(3, col):
        for i in range(filas):
            x = df.iloc[i, j]
            df.iloc[i, j] = puntos_por_comas(x)




editar_lgtd_onda_columnas()

editar_lgtd_onda_valores()



df.to_csv(full_path, sep=';', index=False)





