# getting the libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# getting the data
data = pd.read_csv('financial_data.csv')

# discovering the data
print(data.head())
print(data.describe())
print(data.isna().any())

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

# One hot encoding
X = pd.get_dummies(X)

# making standerdization
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_std = std.fit_transform(X)

# Doing dimensionality reduction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 22)
pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

pca = PCA(n_components = 2)
reduced = pca.fit_transform(X_std)

# ploting 2D data
plt.scatter(reduced[:,0], reduced[:,1], c = y)
plt.show()

plt.scatter(reduced[:100,0], reduced[:100,1], c = y[:100])
plt.show()

# train split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_std, y, 
                                                    test_size = 0.3)

# split without standerdize data
x_train_n, x_test_n, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3)

# split the reduced data
x_train_r, x_test_r, y_train, y_test = train_test_split(reduced, y, 
                                                    test_size = 0.3)

# function for evaluation
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, confusion_matrix

result = pd.DataFrame(columns = ['Model', 'Accuracy' , 'F-one', 'F-beta'])

def evaluate(clf, train = False):
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fbeta = fbeta_score(y_test, y_pred, beta = 0.5)
    # cm = confusion_matrix(y_test, y_pred)
    result.append(pd.DataFrame([[type(clf).__name__ , accuracy, f1, fbeta]], columns = ['Model', 'Accuracy' , 'F-one', 'F-beta']) , ignore_index = True)
    print(result.head())
    print("Test Scores for {} : \nThe Accuracy: {}\nF1: {} \nFbeta: {}".format(type(clf).__name__, accuracy, f1, fbeta))
    if train:
        y_pred = clf.predict(x_train)
        accuracy = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        fbeta = fbeta_score(y_train, y_pred, beta = 0.5)
        # cm = confusion_matrix(y_train, y_pred)
        print("Train Scores for {} : \nThe Accuracy: {}\nF1: {} \nFbeta: {}".format(type(clf).__name__, accuracy, f1, fbeta))    
    return accuracy , f1, fbeta


# start building ML Model's
from sklearn.svm import SVC        
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
###########################
clf_svm = SVC(C = 10, kernel= 'linear')
clf_svm.fit(x_train, y_train)
evaluate(clf_svm, result)
###########################
clf_lr = LogisticRegression()
clf_lr.fit(x_train, y_train)
evaluate(clf_lr)
###########################
clf_nb = GaussianNB()
clf_nb.fit(x_train, y_train)
evaluate(clf_nb)
###########################
clf_dt = DecisionTreeClassifier()
clf_dt.fit(x_train, y_train)
evaluate(clf_dt)
###########################
clf_rf = RandomForestClassifier(n_estimators = 100)
clf_rf.fit(x_train, y_train)
evaluate(clf_rf)
###########################
clf_ab = AdaBoostClassifier()
clf_ab.fit(x_train, y_train)
evaluate(clf_ab)
###########################
clf_bg = BaggingClassifier()
clf_bg.fit(x_train, y_train)
evaluate(clf_bg)
###########################
clf_knn = KNeighborsClassifier(n_neighbors = 100)
clf_knn.fit(x_train, y_train)
evaluate(clf_knn)