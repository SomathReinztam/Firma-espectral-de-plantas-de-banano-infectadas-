"""
Lasso_LinearRegression_Tuning_Hyperparameters.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

for alpha in [0.01, 0.1, 1, 10 , 100]:
    clf = Lasso(alpha=alpha, max_iter=4000)
    clf.fit(X_train, y_train)
    y_test_predic = clf.predict(X_test)
    score = mean_absolute_error(y_test, y_test_predic)

    print('score:', score)
    print('alpha:', alpha)
    print()

    if score < best_score:
        best_score = score
        best_parameter = alpha

print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameter))

