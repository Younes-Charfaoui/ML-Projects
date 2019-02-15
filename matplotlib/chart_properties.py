import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10)
y = x ^ 2
z = x ^ 3
t = x ^ 4
plt.xlabel('Distance')
plt.ylabel('Time')
plt.title("Graph Drawing")
plt.plot(x , y)
plt.show()

## anotating a graph
plt.xlabel('Distance')
plt.ylabel('Time')
plt.title("Graph Drawing")
plt.plot(x , y)
plt.annotate(xy=[2,1], s = "Second Entry")
plt.annotate(xy=[4,6], s = "Third Entry")
plt.show()

## Adding legend
plt.xlabel('Distance')
plt.ylabel('Time')
plt.title("Graph Drawing")
plt.plot(x, y)
plt.plot(x, z)
plt.plot(x, t)
plt.legend(['Race One','Race Two','Race Three'] , loc = 4)
plt.show()

## adding style
plt.xlabel('Distance')
plt.ylabel('Time')
plt.title("Graph Drawing")
plt.plot(x, y)
plt.plot(x, z)
plt.plot(x, t)
plt.legend(['Race One','Race Two','Race Three'] , loc = 4)
plt.style.use('fivethirtyeight')
plt.show()
## to see available
print(plt.style.available)