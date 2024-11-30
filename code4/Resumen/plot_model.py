"""
plot_clf_model.py

"""

import os 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_datos(random_state, etiqueta):
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
    path = path.replace("\\", "/")

    file = 'datos.csv'
    file = os.path.join(path, file)

    df = pd.read_csv(file, sep=';')
    X_df = df.iloc[:, 4:].values
    y_df = df[etiqueta].values

    X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.3,
    random_state=random_state
    )   

    return X_test, y_test



def transfrom_data(X_data):
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code4\Sets"
    path = path.replace('\\', '/')

    file = 'model_PCA_2d.pkl'
    file = os.path.join(path, file)

    with open(file, 'rb') as f:
        pca = pickle.load(f)
    
    return pca.transform(X_data)


def fun(y_predict, y_true):
             #   Predict  True
    a = []   #      0      0
    b = []   #      1      1
    c = []   #      0      1
    d = []   #      1      0

    for i in range(len(y_true)):
        if (y_predict[i] == 0) and (y_true[i] == 0):
            a.append(i)
        if (y_predict[i] == 1) and (y_true[i] == 1):
            b.append(i)
        if (y_predict[i] == 0) and (y_true[i] == 1):
            c.append(i)
        if (y_predict[i] == 1) and (y_true[i] == 0):
            d.append(i)
    return a, b, c, d

def plot_clf_model(clf, X_data, y_data):

    print('score:', clf.score(X_data, y_data))

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    d = {
        0:['#16EB00', '^', 'TE'],
        1:['#16EB00', 'o', 'TS'],
        2:['#FE0006', '^', 'FE'],
        3:['#FE0006', 'o', 'FS']
    }
    
    y_predict = clf.predict(X_data)
    X_trasform = transfrom_data(X_data)

    L = fun(y_predict, y_data)
    y_predict = None

    t = 0
    for lista in L:
        x = []
        y = []
        for i in lista:
            x.append(X_trasform[i, 0])
            y.append(X_trasform[i, 1])
        
        ax.scatter(x, y, color=d[t][0], marker=d[t][1], label=d[t][2])
        t += 1
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()






def plot_reg_model(reg, X_data, y_data):
    
    y_data_predict = reg.predict(X_data)

    print(
        'data mean_absolute_error:',
        mean_absolute_error(y_data, y_data_predict)
    )

    a = [(y_data[i], y_data_predict[i]) for i in range(len(y_data))]
    a_ordenada = sorted(a, key=lambda x: x[0])

    y1 = [a_ordenada[i][0] for i in range(len(a_ordenada))]
    y2 = [a_ordenada[i][1] for i in range(len(a_ordenada))]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.plot(y1, label='data', marker='o', linestyle='None')
    ax.plot(y2, label='predict', marker='o', linestyle='None')

    ax.set(xlabel='Samples', ylabel='dpi: dias', title='dpi en test')

    plt.legend()
    plt.show()