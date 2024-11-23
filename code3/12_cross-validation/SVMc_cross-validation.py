"""
SVMc_cross-validation.py   

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)



def data_split():
    df = pd.read_csv(file, sep=';')

    X_df = df.iloc[:, 4:].values
    y_df = df['Sana'].values
    df = None

    X_trian, X_test, y_train, y_test = train_test_split(
        X_df,
        y_df,
        test_size=0.2,
        random_state=81
    )
    return X_trian, X_test, y_train, y_test



X_trian, X_test, y_train, y_test = data_split()

hparams = {'C': 0.3020408163265306, 'kernel':'linear'}
clf = SVC(**hparams)

kf = KFold(n_splits=6)

cv_scores = cross_val_score(clf, X_trian, y_train, cv=kf)

print('Se obtienen los siguientes coeficientes de determinación:')
print(cv_scores, '\n')
print(f'Max R-Squared: {max(cv_scores)}')
print(f'Min R-Squared: {min(cv_scores)}')
print('Promedio R-Squared: {:.3f}'.format(np.mean(cv_scores)))
print('Desviación Estándar: {:.3f}'.format(np.std(cv_scores)))
print(f'Intervalo de confianza 95%: {np.quantile(cv_scores, [0.025, 0.975])}')

print()

clf.fit(X_trian, y_train)
score_test = clf.score(X_test, y_test)
print('score en test:', score_test)




