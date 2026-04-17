import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [10,20,25,30]

plt.bar(x,y)
plt.title("Bar")
plt.show()

plt.plot(x,y)
plt.title("Line")
plt.show()

plt.scatter(x,y)
plt.title("Scatter")
plt.show()