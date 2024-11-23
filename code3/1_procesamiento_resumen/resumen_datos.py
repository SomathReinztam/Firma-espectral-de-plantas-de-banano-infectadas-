"""
resumen_datos.py


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

df2 = df.describe()

df2.to_csv(full_path2, sep=';', index=True)

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
    print(df2)
    print("\nClases diferentes en la columna dpi:")
    print(df['dpi'].unique())
    print("\nClases diferentes en la columna Sana:")
    print(df['Sana'].unique())
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
    print("\nClases diferentes en la columna Planta:")
    print(df['Planta'].unique())

resumen_df(df)

