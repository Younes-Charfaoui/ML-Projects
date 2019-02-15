import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.rand(10,5) , 
                  columns = ['A', 'B', 'C', 'D', 'E'])
df.plot.box(True)

## heat map
data=[{2,3,4,1},{6,3,5,2},{6,3,5,4},{3,7,5,4},{2,8,1,5}]
Index= ['I1', 'I2','I3','I4','I5']
Cols = ['C1', 'C2', 'C3','C4']
df2 = pd.DataFrame(data , index = Index , columns = Cols)

plt.pcolor(df)
plt.show()

## scatter plot
df3 = pd.DataFrame(np.random.rand(50,4) , columns= ['a', 'b', 'c', 'd'])
df3.plot.scatter(x = 'a' , y = 'b')
plt.scatter(df3['a'] , df3['b'])
plt.show()

## bubble chart
x = np.random.rand(40)
y = np.random.rand(40)
z = np.random.rand(40)
colors = np.random.rand(40)
plt.scatter(x , y , s = z * 1000, c = colors)
plt.show()

## 3d plot
from mpl_toolkits.mplot3d import axes3d

chart = plt.figure()
chart_3d = chart.add_subplot(111,projection='3d')
X,Y,Z = axes3d.get_test_data(0.08)
chart_3d.plot_wireframe(X,Y,Z,color='r',rstride=15, cstride=10)
plt.show()