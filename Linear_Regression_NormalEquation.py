
import numpy as np


def findTheta(X, Y, features, no_of_training_samples):
	X = np.array(X)
	Y = np.array(Y).reshape(no_of_training_samples, 1)
	x_transpose = np.transpose(X)
	theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_transpose, X)), x_transpose), Y)
	return theta


x = [[1,2,3],
	 [1,8,2],
	 [1,11,12]]

y = [5,10,23]

features = 3
no_of_training_samples = 3

print(findTheta(x, y, features, no_of_training_samples))