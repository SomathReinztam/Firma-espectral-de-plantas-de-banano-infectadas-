"""
plot_1.py

"""
import os
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_path = os.path.join(path, file)

df = pd.read_csv(full_path, sep=';')


# -----------


def ver_mean_std(filtro):
    df_filtro = df[filtro]
    n = df_filtro.shape[0]
    lgtd_onda = [float(i) for i in df_filtro.columns.values[4:]]

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1) = plt.subplots(nrows=2,  figsize=(12, 6))

    for i in range(n):
        y = df_filtro.iloc[i, 4:]
        ax0.plot(lgtd_onda, y, color='#99B098')
    ax0.plot(lgtd_onda, y, color='#99B098', label='sample')
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax0.plot(lgtd_onda, y, color='#CF0079', linewidth=2.5, label='mean')
    ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

    y = df_filtro.iloc[:, 4:].std(axis=0).values
    ax1.plot(lgtd_onda, y, color='#32AFA2', linewidth=2.5, label='std')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()




def ver_mean_Contro_dpi():
    dpi_clases = [0, 7, 14, 21]
    n = len(dpi_clases)
    lgtd_onda = [float(i) for i in df.columns.values[4:]]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    for i in range(n):
        dpi = dpi_clases[i]
        filtro = (df['Tratamiento'] == 0) & (df['dpi'] == dpi)
        df_filtro = df[filtro]
        y = df_filtro.iloc[:, 4:].mean(axis=0).values
        ax.plot(lgtd_onda, y, color='#33EE15', linewidth=1.5)
    ax.plot(lgtd_onda, y, color='#33EE15', linewidth=1.5, label="Con")

    filtro = (df['Tratamiento'] == 0)
    df_filtro = df[filtro]
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax.plot(lgtd_onda, y, color='#4B2DE0', linewidth=2.5, label="Mean Con")

    E = 7
    filtro = (df['Tratamiento'] == 1) & (df['dpi'] == E)
    df_filtro = df[filtro]
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax.plot(lgtd_onda, y, color='#FFEF00', linewidth=1.5, label="No Sana")

    E = 21
    filtro = (df['Tratamiento'] == 1) & (df['dpi'] == E)
    df_filtro = df[filtro]
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax.plot(lgtd_onda, y, color='#FF8700', linewidth=1.5, label="No Sana")

    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend()
    plt.show()





def ver_meanControl_vs_Tratamiento_en_dpi_0_mean():
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    Tratamiento_class = [1, 2, 3, 4, 5]
    n = len(Tratamiento_class)
    s = ['-', '--', '-.', ':', '-']
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    for i in range(n):
        k = Tratamiento_class[i]
        filtro = (df['Tratamiento'] == k) & (df['dpi'] == 0)
        df_filtro = df[filtro]
        y = df_filtro.iloc[:, 4:].mean(axis=0).values
        ax.plot(lgtd_onda, y, color='#03AFAF', linewidth=1.5, linestyle=s[i], label="{0} 0 Mean".format(d[k]))
    
    filtro = (df['Tratamiento'] == 0) 
    df_filtro = df[filtro]
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax.plot(lgtd_onda, y, color='#85F200', linewidth=2.5, label="Con Mean")

    plt.legend()
    plt.show()

def ver_meanControl_vs_Tratamiento_en_dpi_mean(dpi):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    Tratamiento_class = [1, 2, 3, 4, 5]
    n = len(Tratamiento_class)
    s = ['-', '--', '-.', ':', '-']
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    for i in range(n):
        k = Tratamiento_class[i]
        filtro = (df['Tratamiento'] == k) & (df['dpi'] == dpi)
        df_filtro = df[filtro]
        y = df_filtro.iloc[:, 4:].mean(axis=0).values
        ax.plot(lgtd_onda, y, color='#03AFAF', linewidth=1.5, linestyle=s[i], label="{0} 0 Mean".format(d[k]))
    
    filtro = (df['Tratamiento'] == 0) 
    df_filtro = df[filtro]
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax.plot(lgtd_onda, y, color='#85F200', linewidth=2.5, label="Con Mean")

    plt.legend()
    plt.show()
    

def ver_plantas(filtro):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    s = ['-', '--', '-.', ':', '-']

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    df_filtro = df[filtro]
    m = df_filtro.shape[0]

    for i in range(m):
        serie = df_filtro.iloc[i, :]
        dpi = int(serie['dpi'])
        T = int(serie['Tratamiento'])
        T = d[T]
        Planta = int(serie['Planta'])
        y = serie[4:].values
        
        ax.plot(lgtd_onda, y, linewidth=1.5, label="dpi:{0} {1} P:{2}".format(dpi, T, Planta))
    

    plt.legend()
    #plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1))
    plt.show()




    


#filtro = (df['Tratamiento'] == 1) & (df['dpi'] == 21)
#filtro = (df['Sana'] == 0)
#ver_mean_std(filtro)

# -------

#ver_mean_Contro_dpi()


# -------

#ver_meanControl_vs_Tratamiento_en_dpi_0_mean()


# ------------

#dpi = 0
#ver_meanControl_vs_Tratamiento_en_dpi_mean(dpi)


# ------------

#filtro = (df['dpi'] == 21) & (df['Planta'] == 5) & ((df['Tratamiento'] == 1) | (df['Tratamiento'] == 2) | (df['Tratamiento'] == 0)) 

#filtro = (df['Tratamiento'] == 3) & (df['Planta'] == 17)
#ver_plantas(filtro)







