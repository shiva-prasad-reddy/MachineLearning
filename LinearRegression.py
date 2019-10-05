
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")


class LinearRegression():
	"""Univariate Linear Reression"""

	p0 = []
	p1 = []
	j = []

	def __init__(self, x_train, y_train, param_0 = 0.0, param_1 = 0.0):
		self.X = np.array(x_train)
		self.Y = np.array(y_train)

		self.M = self.X.size

		self.param_0 = param_0
		self.param_1 = param_1


	def fit(self):
		self.H = self.param_0 + (self.param_1 * self.X)
		plt.plot(self.X, self.H, linewidth='0.5')


	def cost(self):
		self.J = np.sum((self.H - self.Y) ** 2) / (2 * self.M)
		return self.J

	def minimizeCost(self, alpha=0.05):
		"""alpha -- learning rate"""
		
		LinearRegression.p0.append(self.param_0)
		LinearRegression.p1.append(self.param_1)
		LinearRegression.j.append(self.J)

		der_param_0 = np.sum((self.H - self.Y)) / self.M
		der_param_1 = np.dot((self.H - self.Y), self.X) / self.M
		
		#correct simultaneous update
		temp_0 = self.param_0 - (alpha * der_param_0)
		temp_1 = self.param_1 - (alpha * der_param_1)
		
		self.param_0 = temp_0
		self.param_1 = temp_1

	def graph(self):
		plt.scatter(self.X, self.Y, color='r', marker='x')
		plt.xlabel('X')
		plt.ylabel('H')
		plt.title("Linear Reression")
		#plt.legend()
		plt.show()


	def predict(self, test_x):
		test_x = np.array(test_x)
		output = self.param_0 + (self.param_1 * test_x)
		return output


	def graphCost(self):
		fig = plt.figure()
		axes = fig.add_subplot(111, projection='3d')
		
		LinearRegression.p0 = np.array(LinearRegression.p0)
		LinearRegression.p1 = np.array(LinearRegression.p1)
		LinearRegression.j = np.array(LinearRegression.j)

		x,y = np.meshgrid(LinearRegression.p0, LinearRegression.p1)
		z = LinearRegression.j.reshape(x.shape[0],1)
		
		axes.plot_surface(x, y, z, cmap='viridis')
		axes.set_xlabel("param_0")
		axes.set_ylabel("param_1")
		axes.set_zlabel("J(param_0, param_1)")

		plt.show()






x = [1,2,3]
y = [1,2,3]

LR = LinearRegression(x,y)

for i in range(500):
	LR.fit()
	cost = LR.cost()
	print(cost)
	LR.minimizeCost()

print("Prediction ---> \n", LR.predict([1,2,3,4,5,6]))
LR.graphCost()


