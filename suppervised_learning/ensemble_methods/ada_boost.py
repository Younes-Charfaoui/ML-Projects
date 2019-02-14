# -*- coding: utf-8 -*-

import pandas as pd
data = pd.read_csv('../svm/data.csv' , header = None)
x_values = data.iloc[:,:-1].values
y_values = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts = train_test_split(x_values , y_values,test_size = 0.2)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 2) , n_estimators = 10)
model.fit(x_tr, y_tr)
y_pred = model.predict(x_ts)

from sklearn.metrics  import accuracy_score
score = accuracy_score(y_ts , y_pred)
print("The score is ", score)

