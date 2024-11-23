"""
Plot_LinearRegression.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from numpy import abs

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos_no_control.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

X_df = df.iloc[:, 4:].values
y_df = df['dpi'].values

df = None

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.3,
    random_state=203
)

X_df, y_df = None, None

reg = LinearRegression()
reg.fit(X_train, y_train)


y_predic = reg.predict(X_train)
y_predic_test = reg.predict(X_test)
print()
print('mean_absolute_error train:' ,mean_absolute_error(y_train, y_predic))
print('r2_score train:', r2_score(y_train, y_predic))
print('mean_squared_error train:', mean_squared_error(y_train, y_predic))

print()
print('mean_absolute_error test:', mean_absolute_error(y_test, y_predic_test))
print('r2_score test:', r2_score(y_test, y_predic_test))
print('mean_squared_error test:', mean_squared_error(y_test, y_predic_test))



def plot_dpi_test():
    y_predic_test = reg.predict(X_test)
    a = [(y_test[i], y_predic_test[i]) for i in range(len(y_test))]
    a_ordenada = sorted(a, key=lambda x: x[0])

    y1 = [a_ordenada[i][0] for i in range(len(a_ordenada))]
    y2 = [a_ordenada[i][1] for i in range(len(a_ordenada))]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.plot(y1, label='data', marker='o', linestyle='None')
    ax.plot(y2, label='predic', marker='o', linestyle='None')

    ax.set(xlabel='Samples', ylabel='dpi: dias', title='dpi en test')

    plt.legend()
    plt.show()




def plot_dpi_train():
    y_predic_train = reg.predict(X_train)
    a = [(y_train[i], y_predic_train[i]) for i in range(len(y_train))]
    a_ordenada = sorted(a, key=lambda x: x[0])

    y1 = [a_ordenada[i][0] for i in range(len(a_ordenada))]
    y2 = [a_ordenada[i][1] for i in range(len(a_ordenada))]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.plot(y1, label='data', marker='o', linestyle='None')
    ax.plot(y2, label='predic', marker='o', linestyle='None')

    plt.legend()
    plt.show()





def plot_predic_vs_truth_test():
    y_predic_test = reg.predict(X_test)

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    y = y_test - y_predic_test
    y = abs(y)

    ax.plot(y, label='error abs', linestyle='-')

    plt.legend()
    plt.show()



plot_dpi_test()
#plot_dpi_train()
#plot_predic_vs_truth_test()

