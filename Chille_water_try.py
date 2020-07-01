from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from fbprophet import Prophet


chilled_water=pd.read_csv('chilled_water.csv')
reg=LinearRegression()
del chilled_water['primary_use']
print(chilled_water)
y=chilled_water['chilled water'].to_numpy()
del chilled_water['chilled water']
X=chilled_water.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size = 0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
print(reg_all.score(X_test, y_test))



