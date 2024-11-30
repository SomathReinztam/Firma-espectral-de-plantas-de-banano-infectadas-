import os
import pandas as pd
from learning_Curve import plot_clf_learning_Curve, plot_reg_learning_Curve

from sklearn.svm import SVC
from sklearn.linear_model import Ridge

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')


"""
hparams = {'C': 0.3020408163265306, 'kernel':'linear'}
clf = SVC(**hparams)
scoring = 'accuracy'
plot_clf_learning_Curve(clf, df,scoring)
"""

"""

"""
reg = Ridge(7)
scoring='neg_mean_absolute_error'
plot_reg_learning_Curve(reg, df, scoring)


