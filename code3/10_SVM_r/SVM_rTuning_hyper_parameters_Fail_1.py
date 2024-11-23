"""
SVM_rTuning_hyper_parameters_Fail_1.py


FAIL muy pesado para mi PC
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

import pickle

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
    random_state=79
)

X_df = None
y_df = None

reg = SVR()



param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.01, 0.1, 1, 10, 25],
    'gamma': ['scale', 0.5, 1, 10, 25],
    'coef0': [-5, 0, 1, 15],
    'epsilon': [0.1, 1, 10],
    'degree': [1, 3, 6]
}


grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)


# Fail

# Ajustamos el modelo en el conjunto de entrenamiento
grid_search.fit(X_train, y_train)

# Obtenemos los mejores parámetros y el mejor score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Mejores hiperparámetros:", best_params)
print("Mejor puntaje (neg_mean_squared_error):", best_score)

file = 'mejor_modelo_svr.pkl'
file = os.path.join(path, file)

with open(file, 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)