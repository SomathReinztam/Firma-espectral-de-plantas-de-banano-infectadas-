"""
procesamiento.py

El objetivo de este script es trasformar los datos float
de las columnas dpi y Planta a int, y trasformar los 
datos categoricos de Tratamiento a enteros
(label encoder)

Posteriormente se guradara este nuevo data set
bajo el nobre de datos.csv
"""

import os
import pandas as pd

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "resumen_aislados.csv"
full_path = os.path.join(path, file)

new_file = "datos.csv"
new_full_path = os.path.join(path, new_file)

df = pd.read_csv(full_path, sep=';')

# Eliminar las filas completamente vacías
df.dropna(how='all', inplace=True)

# Eliminar las columnas completamente vacías
df.dropna(axis=1, how='all', inplace=True)


def procesar_dpi_Planta():
    df['dpi'] = df['dpi'].astype(int)
    df['Planta'] = df['Planta'].astype(int)


def procesar_Tratamiento():
    d = [('Con', 0), ('Fo5', 1), ('IsB', 2), ('Var', 3), ('ViE', 4), ('HyS', 5)]
    d = dict(d)
    lista = []
    n = df.shape[0]
    for i in range(n):
        x = df.iloc[i, 1]
        try:
            x = d[x]
            df.iloc[i, 1] = x
        except:
            lista.append(i)
    df['Tratamiento'].astype(int)
    return lista

def resumen_df(df):
    print(df.head())
    print()
    print("Número de filas y columnas:", df.shape)
    print("\nTipos de variables:")
    print(df.dtypes)
    print("\nConteo de valores nulos por columna:")
    print(df.isnull().sum())
    print("\nNúmero total de valores nulos:", df.isnull().sum().sum())
    print("\nPorcentaje de valores nulos por columna:")
    print(df.isnull().mean() * 100)
    print("\nDescripción estadística (numérica):")
    print(df.describe())
    print("\nClases diferentes en la columna dpi:")
    print(df['dpi'].unique())
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
    print("\nClases diferentes en la columna Planta:")
    print(df['Planta'].unique())


print(procesar_dpi_Planta())
print()
print(procesar_Tratamiento())
print()



resumen_df(df)



df.to_csv(new_full_path, sep=';', index=False)
