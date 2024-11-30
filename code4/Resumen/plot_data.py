"""
plot_data.py

"""

import os
import pandas as pd
import data_funtions as datf

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

file2 = 'datos_PCA_2d.csv'
file2 = os.path.join(path, file2)

file3 = 'datos_tSNE-2d.csv'
file3 = os.path.join(path, file3)

file4 = 'datos_describe.csv'
file4 = os.path.join(path, file4)

df = pd.read_csv(file, sep=';')
#df2 = pd.read_csv(file2, sep=';')
#df3 = pd.read_csv(file3, sep=';')
#df4 = pd.read_csv(file4, sep=';')

"""
Tratamiento = 5
Planta = 21
ver_planta(df,Tratamiento, Planta)

"""



"""
a = ((df['dpi'] == 7) & (df['Planta'] == 17) & (df['Tratamiento'] == 3))
b = ((df['dpi'] == 21) & (df['Planta'] == 9) & (df['Tratamiento'] == 5))
filtro = a | b
ver_plantas_filtro(df, filtro)

"""



"""
filtro = (df['dpi'] == 7) & (df['Tratamiento'] == 3)
ver_mean_std(df, filtro)

"""


"""
filtros = [
    (df['dpi'] == 0) & (df['Tratamiento'] == 0),
    (df['dpi'] == 0) & (df['Tratamiento'] == 1),
    (df['dpi'] == 0) & (df['Tratamiento'] == 5)
]
ver_Means_Stds(df, filtros)

"""


"""
filtros = [
    (df['dpi'] == 0) & (df['Tratamiento'] != 0),
    (df['Tratamiento'] == 0)
]
ver_Means_Stds_free(df, filtros)

"""

# -------------------------------------

"""
datf.plot_datos_PCA_2d(df2)
"""

"""
datf.plot_datos_sanos_PCA_2d(df2)
"""

"""
datf.plot_datos_Tratamiento_VS_dpi_PCA2d(df2, Tratamiento=2)
"""


# ---------------------------


"""
datf.plot_datos_tSNE_2d(df3)
"""

"""
datf.plot_datos_Tratamiento_VS_dpi_tSNE_2d(df3, Tratamiento=4)
"""


# ----------------------------------------------------


#datf.resumen_datos(df)

#datf.plot_describe(df4)

#datf.plot_describe_2(df4)

# ----------------------------------------------------


df = df.iloc[:, 4:]
datf.plot_heat_map(df)