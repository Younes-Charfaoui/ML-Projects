import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10)
y = x ** 2
plt.plot(x , y)
plt.show()

## labeling the axes and color
plt.xlabel('The X')
plt.ylabel('The Y')
plt.title("Squared Function")
plt.plot(x , y , '>')
plt.plot(x , y , 'r')
plt.show()

## saving the file
plt.savefig('squared.pdf' , format='pdf')