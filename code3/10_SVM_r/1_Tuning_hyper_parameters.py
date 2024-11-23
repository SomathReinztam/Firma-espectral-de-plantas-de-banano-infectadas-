"""
1_Tuning_hyper_parameters.py

Tarda demaciado !!!
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import time

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
    random_state=93
)

X_df = None
y_df = None

print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', X_test.shape)
print('y_test size:', y_test.shape)
print()

kernel = 'linear'
best_metric = 0

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    start_time = time.time()
    reg = SVR(kernel=kernel, C=C)
    reg.fit(X_train, y_train)
    y_test_predic = reg.predict(X_test)
    metric = mean_absolute_error(y_test, y_test_predic)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('metric:', metric)
    print('C:', C)
    print('Elapsed time (seconds):', elapsed_time)
    print()
    if metric > best_metric:
        best_metric = metric
        best_parameters = {
                'kernel':kernel,
                'C':C,
            }
        
print()
print("Best score: {:.2f}".format(best_metric))
print("Best parameters: {}".format(best_parameters))