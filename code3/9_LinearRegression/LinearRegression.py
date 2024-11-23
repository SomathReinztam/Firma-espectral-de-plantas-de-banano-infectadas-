"""
LinearRegression.py   

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

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

reg = LinearRegression()
reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

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

#y1 = y_test - y_predic_test
#y2 = y_train - y_predic

plt.style.use('seaborn-whitegrid')
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 8))

ax0.plot(y_test, label='data', marker='o', linestyle='None')
ax0.plot(y_predic_test, label='predic', marker='o', linestyle='None')
ax0.legend()

ax1.plot(y_train, label='data', marker='o', linestyle='None')
ax1.plot(y_predic, label='predic', marker='o', linestyle='None')
ax1.legend()

plt.show()