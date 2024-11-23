"""
Ridge_LinearRegression_Tuning_Hyperparameters.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import numpy as np

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

X_df, y_df = None, None




best_score = math.inf
for alpha in np.linspace(7, 8, 10):
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    y_test_predic = reg.predict(X_test)
    score = mean_absolute_error(y_test, y_test_predic)

    print('score:', score)
    print('alpha:', alpha)
    print()

    if score < best_score:
        best_score = score
        best_parameters = alpha
    

print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameters))