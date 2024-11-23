"""
SVMc_Learning_Curve.py   

"""

import os
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, learning_curve
import numpy as np

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')
X_df = df.iloc[:, 4:].values
y_df = df['Sana'].values

df = None



hparams = {'C': 0.3020408163265306, 'kernel':'linear'}
clf = SVC(**hparams)

train_sizes, train_scores, test_scores = learning_curve(
    clf,
    X_df,
    y_df,
    cv=8,
    scoring='accuracy',
    n_jobs=-1,
    shuffle=True,
    random_state=97,
)

# Calcular promedios y desviaciones estándar
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Graficar la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training score", color="blue")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.2)
plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="orange")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="orange", alpha=0.2)

# Etiquetas y título
plt.title("Learning Curve: SVC with linear kernel", fontsize=16)
plt.xlabel("Training examples", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(True)
plt.show()



"""
display = LearningCurveDisplay(train_sizes=train_sizes,
    train_scores=train_scores, test_scores=test_scores, score_name="Score")

display.plot()
plt.show()

"""


