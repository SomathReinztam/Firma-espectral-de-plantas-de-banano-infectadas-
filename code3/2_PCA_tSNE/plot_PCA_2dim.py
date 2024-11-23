"""
plot_PCA_2dim.py

"""

import os
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos_PCA_2d.csv"
full_path = os.path.join(path, file)

df = pd.read_csv(full_path, sep=';')

def plot_1_datosPCA_2d():
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    estilos = ['x' ,'d', '^', 's', '+']
    colores = ['#2F900B', '#2A9B00', '#3BDB00', '#4FD51E', '#2A9B00']
    Tratamiento_class = [1, 2, 3, 4, 5]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    n = len(Tratamiento_class)

    filtro = (df['Sana'] == 0)
    df_filtro = df[filtro]
    x = df_filtro['eje_x']
    y = df_filtro['eje_y']
    ax.scatter(x, y, color='#FF7800', marker='o', label="No Sana")

    filtro = (df['Tratamiento'] == 0) & (df['dpi'] == 0)
    df_filtro = df[filtro]
    x = df_filtro['eje_x']
    y = df_filtro['eje_y']
    ax.scatter(x, y, color='#9FF700', marker='o', label="Con")


    for i in range(n):
        Tc = Tratamiento_class[i]
        filtro = (df['Tratamiento'] == Tc) & (df['dpi'] == 0)
        df_filtro = df[filtro]
        x = df_filtro['eje_x']
        y = df_filtro['eje_y']
        c = colores[i]
        s = estilos[i]
        ax.scatter(x, y, color=c, marker=s, label="{0}".format(d[Tc]))
    


    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def plot_2_datosPCA_2d():
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    estilos = ['x' ,'d', '^', 's', '+']
    colores = ['#2F900B', '#2A9B00', '#3BDB00', '#4FD51E', '#2A9B00']
    Tratamiento_class = [1, 2, 3, 4, 5]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    n = len(Tratamiento_class)

    filtro = (df['Tratamiento'] == 0) & (df['dpi'] == 0)
    df_filtro = df[filtro]
    x = df_filtro['eje_x']
    y = df_filtro['eje_y']
    ax.scatter(x, y, color='#9FF700', marker='o', label="Con")


    for i in range(n):
        Tc = Tratamiento_class[i]
        filtro = (df['Tratamiento'] == Tc) & (df['dpi'] == 0)
        df_filtro = df[filtro]
        x = df_filtro['eje_x']
        y = df_filtro['eje_y']
        c = colores[i]
        s = estilos[i]
        ax.scatter(x, y, color=c, marker=s, label="{0}".format(d[Tc]))
    


    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()





def plot_Tratamiento_atravez_tiempo(Tratamiento):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    dpi_class = [0, 7, 14, 21]
    n = len(dpi_class)
    estilos = ['x' ,'d', '^', 's', '+']
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    colores = ['#FF0000', '#D6007D', '#2A05C1', '#A1F200']
    
    for i in range(n):
        dpi = dpi_class[i]
        filtro = (df['Tratamiento'] == Tratamiento) & (df['dpi'] == dpi)
        df_filtro = df[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        c = colores[i]
        s = estilos[i]
        l = "{0} {1}".format(d[Tratamiento], dpi)
        ax.scatter(x, y, color=c, marker=s, label=l)
        
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

#plot_1_datosPCA_2d()

#plot_2_datosPCA_2d()

plot_Tratamiento_atravez_tiempo(Tratamiento=5)
    

