import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""Wrong dataset as it is is suitable for classification"""
my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/Job_Placement_Data.csv")
print(my_file.head())

X_features = my_file[["ssc_percentage","hsc_percentage", "degree_percentage", "emp_test_percentage", "mba_percent"]]
y_features = my_file["status01"]


print(X_features.shape)
print(y_features.shape)

X_train, X_test, y_train, y_test = train_test_split(X_features,y_features,test_size=0.2,random_state=42)

my_model = LinearRegression()
my_model.fit(X_train,y_train)

predictions = my_model.predict(X_test)
print(y_test.head(15))
print(predictions)

mse = mean_squared_error(y_test,predictions)
rmse = np.sqrt(mse)
print(rmse)




