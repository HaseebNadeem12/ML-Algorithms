import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/Data/Suppervised ML Algorithems/Iris flower classification.csv")
my_file = my_file.dropna()
# print(my_file.head(10))

X_features = my_file[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_features = my_file['Species']

X_train,X_test,y_train,y_test = train_test_split(X_features,y_features,test_size=0.2,shuffle=True)
# print(X_train.head())
# print(y_train.head())

SVM_model = SVC(kernel='linear')
SVM_model.fit(X_train,y_train)

prediction = SVM_model.predict(X_test)

"""Comparing"""
print(y_test.head(15))
print(prediction)

"""Accuracy"""
print("Model's accuracy is: ", accuracy_score(y_test,prediction)*100)
print("\nModel's classification report is: ", classification_report(y_test,prediction))
