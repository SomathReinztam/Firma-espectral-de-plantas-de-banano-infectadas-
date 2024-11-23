"""
plot_SVMc_linear.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt


def get_data(df):
    X_df = df.iloc[:, 4:]
    y_df = df['Sana']

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.33, random_state=96)

    return X_test, y_test.values

def predic_label(X_data):
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
    path = path.replace('\\', '/')

    file = 'SVMc_linear_model.pkl'
    file = os.path.join(path, file)

    with open(file, 'rb') as f:
        clf = pickle.load(f)
    
    return clf.predict(X_data)



def transform_data(X_data):
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
    path = path.replace('\\', '/')

    file = 'PCA2d_model.pkl'
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
    

def plot_SVMc_linear(df):

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    X_data, y_true = get_data(df)

    y_predict = predic_label(X_data)

    X_trasform = transform_data(X_data.values)

    X_data = None

    L = fun(y_predict, y_true)

    d = {
        0:['#16EB00', '^', 'TE'],
        1:['#16EB00', 'o', 'TS'],
        2:['#FE0006', '^', 'FE'],
        3:['#FE0006', 'o', 'FS']
    }

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

        

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)


df = pd.read_csv(file, sep=';')

plot_SVMc_linear(df)

