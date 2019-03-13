import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_test_sub = test_data.iloc[:,1:]
ids = test_data.iloc[:,0]

X = train_data.iloc[:,2:]
y = train_data.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Standardization
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
newX = std.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train, y_test = train_test_split(newX, y, test_size = 0.1)

#### Dimontianality Reduction
from sklearn.decomposition import PCA
pca = PCA(300)
reduced = pca.fit_transform(X)
variance = pca.explained_variance_ratio_
pca = PCA(2)
reduced = pca.fit_transform(newX)

plt.scatter(reduced[:,0],reduced[:,1] , c = y)
plt.show()

from sklearn.model_selection import train_test_split
x_train_red, x_test_red, y_train, y_test = train_test_split(reduced, y, test_size = 0.1)
#################################
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
def score(clf , name):
    ypred = clf.predict(x_test)
    ##### Testing set
    accuracy = accuracy_score(y_test, ypred)
    f1 = f1_score(y_test, ypred)
    cm = confusion_matrix(y_test, ypred)
    print('Accuracy for {} was {} and F1 was {}'.format(name, accuracy, f1))
    """print('Confusion Matrix of {} was \n{}'.format(name ,cm) )
    #### Training Set
    y_pred_train = clf.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    cm = confusion_matrix(y_train, y_pred_train)
    print('Accuracy for {} was {} and F1 was {}'.format(name, accuracy, f1))
    print('Confusion Matrix of {} was \n{}'.format(name ,cm) )"""

#####################################
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', C = 100, gamma = 0.0001)
clf.fit(x_train, y_train)
score(clf, 'SVM')
y_sub = clf.predict(X_test_sub)
target_series = pd.Series(y_sub, name= 'target')
df_submit = pd.concat([ids, target_series], axis =1)
df_submit.to_csv("submission_one.csv", index = False)
############################################################
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
score(clf, 'KNN')
###########################################################
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
score(clf, 'Decision Trees')
###########################################################
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf = RandomForestClassifier()
clf.fit(x_train, y_train)
score(clf, 'RandomForest')
###########################################################
clf = AdaBoostClassifier()
clf.fit(x_train, y_train)
score(clf, "AdaBoost")
###########################################################
from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier()
clf.fit(x_train,y_train)
score(clf, 'BaggingClassifier')
########## 0.72
from sklearn.ensemble import BaggingClassifier
clf2 = BaggingClassifier()
clf2.fit(x_train,y_train)
score(clf2, 'BaggingClassifier')
########## 0.84
y_sub = clf2.predict(X_test_sub)
target_series = pd.Series(y_sub, name= 'target')
df_submit = pd.concat([ids, target_series], axis =1)
df_submit.to_csv("submission_two.csv", index = False)
###########################################################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
score(clf, 'Naive Bayes')
###########################################################
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
score(clf, 'Logistic Regression')
###########################################################
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train,y_train)
score(clf, 'SGDClassifier')
###########################################################
##### Grid Search 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
tuned_parameters = [{'kernel': ['rbf', 'poly','sigmoid'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    