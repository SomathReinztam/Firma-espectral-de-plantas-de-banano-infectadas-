"""
Ridge_cross-validation.py  

"""

import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos_no_control.csv'
file = os.path.join(path, file)
from sklearn.model_selection import train_test_split

def data_split():
    df = pd.read_csv(file, sep=';')

    X_df = df.iloc[:, 4:].values
    y_df = df['Sana'].values
    df = None

    X_trian, X_test, y_train, y_test = train_test_split(
        X_df,
        y_df,
        test_size=0.2,
        random_state=17
    )
    return X_trian, X_test, y_train, y_test

X_trian, X_test, y_train, y_test = data_split()

reg = Ridge(alpha=7)
kf = KFold(n_splits=6)

cv_scores = cross_val_score(reg, X_trian, y_train, cv=kf)

print('Se obtienen los siguientes coeficientes de determinación:')
print(cv_scores, '\n')
print(f'Max R-Squared: {max(cv_scores)}')
print(f'Min R-Squared: {min(cv_scores)}')
print('Promedio R-Squared: {:.3f}'.format(np.mean(cv_scores)))
print('Desviación Estándar: {:.3f}'.format(np.std(cv_scores)))
print(f'Intervalo de confianza 95%: {np.quantile(cv_scores, [0.025, 0.975])}')

print()

reg.fit(X_trian, y_train)
score_test = reg.score(X_test, y_test)
print('score en test:', score_test)

print()

y_test_predic = reg.predict(X_test)
mean_abs_error = mean_absolute_error(y_test, y_test_predic)
print('mean_absolute_error en test:', mean_abs_error)
