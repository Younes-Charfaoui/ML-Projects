# Starting with Libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data
data = pd.read_csv('census.csv')

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X = pd.get_dummies(X)
X = X.values
from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
y = label_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train , X_test,y_train ,y_test = train_test_split(X,y,train_size = 0.7,random_state = 30)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train , y_train)

y_hat = clf.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_hat)