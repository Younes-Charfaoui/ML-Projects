import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv',header = None)

X = data.iloc[:,[0,1]].values
y = data.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y , test_size = 0.2)

plt.scatter(X[:,0] , X[:,1] , c = y)
plt.show()


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred , y_test)

print("The accuracy score is {}".format(score))