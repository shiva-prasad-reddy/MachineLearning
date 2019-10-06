import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():

	def __init__(self, x_train, y_train, no_of_parameters, no_of_training_samples, learning_rate=0.05):
		
		self.X = np.transpose(np.array(x_train))
		self.Y = np.array(y_train).reshape(no_of_training_samples, 1)
		

		self.theta = np.zeros([no_of_parameters, 1])

		self.N = no_of_parameters
		self.M = no_of_training_samples
		self.alpha = learning_rate

	def fit(self):
		self.H = np.transpose(np.matmul(np.transpose(self.theta), self.X))
		return self.H

	def cost(self):
		self.J = np.sum((self.H - self.Y) ** 2) / (2 * self.M)
		return self.J

	def minimizeCostGD(self):
		derivatives = np.matmul(self.X, (self.H - self.Y)) / self.M
		self.theta = self.theta - self.alpha * derivatives
		return self.theta

	def predict(self, x_test):
		x_test = np.transpose(np.array(x_test))
		return np.transpose(np.matmul(np.transpose(self.theta), x_test))


#    x0,x1,x2
x = [[1,2,3],
	 [1,8,2],
	 [1,11,12]]

y = [5,10,23]

no_of_parameters = 3
no_of_training_samples = 3

LR = LinearRegression(x, y, no_of_parameters, no_of_training_samples, learning_rate=0.0005)
cost = []
for i in range(300):
	LR.fit()
	cost.append(LR.cost())
	LR.minimizeCostGD()


plt.plot(np.arange(1,301), cost)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()



x = [[1,5,3],
	 [1,10,2],
	 [1,1,82]]

print(LR.predict(x))
print(LR.theta)