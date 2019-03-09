import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

from keras.models import Sequential
from keras.layers.core import Dense


model = Sequential()
model.add(Dense(4,input_dim = 2, activation = 'sigmoid'))
model.add(Dense(8 , activation = 'sigmoid'))
model.add(Dense(1 , activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam' , metrics = ['accuracy'])
model.summary()

history = model.fit(X, y, nb_epoch=50, verbose=0)

# Scoring the model
score = model.evaluate(X, y)
print("\nAccuracy: ", score[-1])

print("\nPredictions:")
print(model.predict_proba(X))