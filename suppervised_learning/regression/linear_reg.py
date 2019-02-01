import pandas as pd


# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv('bmi_life_data.csv') 
data = bmi_life_data.iloc[:,:].values
X = data[:,2]
y = data[:,1]

# Make and fit the linear regression model
from sklearn.linear_model import LinearRegression
bmi_life_model = LinearRegression()
bmi_life_model.fit(X.reshape(-1, 1),y.reshape(-1, 1))

laos_life_exp = bmi_life_model.predict(21.07931)

