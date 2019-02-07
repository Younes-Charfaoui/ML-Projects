import pandas as pd

data = pd.read_csv('titanic_data.csv')

# drop unwanted columns
data = data.drop(['PassengerId' , 'Ticket' , 'Name' , 'Cabin' ],axis = 1)

# drop nan values
data = data.dropna()

X = data.iloc[:,1:].values
y = data.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()
X[:,1] = sex_encoder.fit_transform(X[:,1])
embarked_label_encoder = LabelEncoder()
X[:,-1] = embarked_label_encoder.fit_transform(X[:,-1])

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y , test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred , y_test)

print("The accuracy score is {}".format(score))