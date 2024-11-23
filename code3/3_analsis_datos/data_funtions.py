"""
data_funtions.py

"""


import matplotlib.pyplot as plt

def ver_planta(df, Tratamiento, Planta):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    #     verde_1    verde_2     naranja    rojo
    c = ['#00D700', '#BBE200', '#FF8E00', '#EA1B00']

    filtro = (df['Tratamiento'] == Tratamiento) & (df['Planta'] == Planta)
    df_filtro = df[filtro]
    n = df_filtro.shape[0]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    for i in range(n):
        serie = df_filtro.iloc[i, :]
        dpi = int(serie['dpi'])
        T = d[Tratamiento]
        y = serie[4:].values

        ax.plot(
            lgtd_onda, 
            y, 
            color=c[i], 
            linewidth=1.5,
            label="dpi:{0} {1} P:{2}".format(dpi, T, Planta)
        )
        
    plt.legend()
    plt.show()


def ver_plantas_filtro(df, filtro):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])

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
    plt.show()



def ver_mean_std(df, filtro):
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



"""
filtros son con respecto a dpi y tratamiento

"""
def ver_Means_Stds(df, filtros):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1) = plt.subplots(nrows=2,  figsize=(12, 6))

    for filtro in filtros:
        df_filtro = df[filtro]
        dpi = df_filtro['dpi'].values[0]
        dpi = int(dpi)
        T = df_filtro['Tratamiento'].values[0]
        T = d[int(T)]
        y1 = df_filtro.iloc[:, 4:].mean(axis=0).values
        y2 = df_filtro.iloc[:, 4:].std(axis=0).values

        ax0.plot(lgtd_onda, y1, linewidth=1.5, label="mean dpi:{0} {1}".format(dpi, T))
        ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.plot(lgtd_onda, y2, linewidth=1.5, label="std dpi:{0} {1}".format(dpi, T))
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()




def ver_Means_Stds_free(df, filtros):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1) = plt.subplots(nrows=2,  figsize=(12, 6))

    x = 0
    for filtro in filtros:
        df_filtro = df[filtro]
        y1 = df_filtro.iloc[:, 4:].mean(axis=0).values
        y2 = df_filtro.iloc[:, 4:].std(axis=0).values

        ax0.plot(lgtd_onda, y1, linewidth=1.5, label="mean filtro {0}".format(x))
        ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.plot(lgtd_onda, y2, linewidth=1.5, label="std filtro {0}".format(x))
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        x += 1

    plt.show()


# -----------------------------------------------




def plot_datos_PCA_2d(df):
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





def plot_datos_sanos_PCA_2d(df):
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


def plot_datos_Tratamiento_VS_dpi_PCA2d(df, Tratamiento):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    dpi_class = [0, 7, 14, 21]
    n = len(dpi_class)
    estilos = ['o' ,'d', '^', 'x', '+', 's']
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    colores = ['#0DEF0D', '#E0FE00', '#FFAD00', '#ED1100']
    
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




# --------------------------------------


def plot_datos_tSNE_2d(df):
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






def plot_datos_Tratamiento_VS_dpi_tSNE_2d(df, Tratamiento):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    dpi_class = [0, 7, 14, 21]
    n = len(dpi_class)
    estilos = ['o' ,'s', '^', 'x', '+']
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    colores = ['#0DEF0D', '#E0FE00', '#FFAD00', '#ED1100']
    
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


#  ----------------------------------------------------


def resumen_datos(df):
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
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
    print("\nClases diferentes en la columna Sana:")
    print(df['Sana'].unique())
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
    print(d)
    print("\nClases diferentes en la columna Planta:")
    print(df['Planta'].unique())
    print()
    n = df.shape[0]
    x = df['Sana'].values.sum()
    print("\nPorcentaje de plantas sanas y no sanas:")
    print('Sanas: {:.1f}%'.format((x/n)*100))
    print('No sanas {:.1f}%:'.format((1 - (x/n))*100))
    print("\nPorcentaje de cada clase de tratamiento")
    for i in range(6):
        filtro = (df['Tratamiento'] == i)
        df_filtro = df[filtro]
        x = df_filtro.shape[0]
        p = (x/n)*100
        p = round(p, 1)
        t = d[i]
        print(
            'Porcentage de la clase {0}: {1}%'.format(t, p)
        )
    print("\nPorcentaje de cada clase de tratamiento por cada dpi")
    l = [0, 7, 14, 21]
    for i in range(6):
        t = d[i]
        print('\nPorcentaje de la clase {0}:'.format(t))
        for j in l:
            filtro = (df['Tratamiento'] == i) & (df['dpi'] == j)
            df_filtro = df[filtro]
            x = df_filtro.shape[0]
            p = (x/n)*100
            p = round(p, 1)
            print('{0}% en dpi: {1}'.format(p, j))




#mean_std, minMax, quantils
def plot_describe(df_describe):
    lgtd_onda = [float(i) for i in df_describe.columns.values[5:]]
    L = [(1, 2), (3, 7), (4, 5, 6)]
    d = [(1, 'mean'), (2, 'std'), (3, 'min'), (7, 'max'), (4, '25%'), (5, '50%'), (6,'75%')]
    d = dict(d)

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,  figsize=(10, 6))
    axs = (ax0, ax1, ax2)
    a = 0
    for ax in axs:
        for i in L[a]:
            y = df_describe.iloc[i, 5:]
            ax.plot(lgtd_onda, y, label=d[i])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        a += 1
        
    
    plt.show()



def plot_heat_map(df):
    data = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='plasma')

    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")

    plt.show()
    
            

    

