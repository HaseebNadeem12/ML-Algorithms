from calendar import month

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error,mean_absolute_error,r2_score

my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/placement.csv")
print(my_file.head())
# print(plt.scatter(my_file['cgpa'],my_file['package']))
plt.xlabel("CGPA")
plt.ylabel("PACKAGE")
# plt.show()

X_features = my_file[['cgpa']]  #-> must be a 2D array
y_features = my_file[['package']]

X_train, X_test, y_train, y_test = train_test_split(X_features,y_features,test_size=0.2,random_state=42)

my_model = LinearRegression()
my_model.fit(X_train,y_train)

predictions = my_model.predict(X_test)
# print(pedictions)
plt.scatter(my_file['cgpa'],my_file['package']) #-> scater plot took X and y component
plt.plot(X_test,predictions,color = 'red')
plt.show()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
R2_score = r2_score(y_test,predictions)
rmse = np.sqrt(mse)
print(mae, mse, rmse, r2_score)

"""creating some columns"""
# To treat this data as multiple lenear equation
my_file['new_column'] = np.random.rand(200,1)

#-> Can change the possition of the column
column = my_file.pop('new_column')
my_file.insert(1,'new_col', column)
print(my_file.head())

