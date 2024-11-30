"""
learning_Curve.py   

"""

import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_clf_learning_Curve(clf, df, scoring):
    X_df = df.iloc[:, 4:].values
    y_df = df['Sana'].values

    train_sizes, train_scores, test_scores = learning_curve(
    clf,
    X_df,
    y_df,
    cv=8,
    scoring=scoring,
    n_jobs=-1,
    shuffle=True,
    random_state=97,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    ax.plot(train_sizes, train_scores_mean, label="Training score", color="blue")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.2)
    ax.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="orange")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="orange", alpha=0.2)

    ax.set_title("Learning Curve: SVC with linear kernel", fontsize=16)
    ax.set_xlabel("Training examples", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)

    plt.legend(loc="best", fontsize=12)
    plt.show()



def plot_reg_learning_Curve(clf, df, scoring):
    X_df = df.iloc[:, 4:].values
    y_df = df['dpi'].values

    train_sizes, train_scores, test_scores = learning_curve(
    clf,
    X_df,
    y_df,
    cv=8,
    scoring=scoring,
    n_jobs=-1,
    shuffle=True,
    random_state=97,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    ax.plot(train_sizes, train_scores_mean, label="Training score", color="blue")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.2)
    ax.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="orange")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="orange", alpha=0.2)

    ax.set_title("Learning Curve: Ridge Linear Reg", fontsize=16)
    ax.set_xlabel("Training examples", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)

    plt.legend(loc="best", fontsize=12)
    plt.show()


