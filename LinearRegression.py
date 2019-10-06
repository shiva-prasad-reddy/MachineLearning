import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
style.use("ggplot")

def fit(x,y):
    global bias, weight
    x = np.array(x)
    
    #x = x / np.max(x) --- feature scaling
    
    y = np.array(y)
    h = bias + (weight * x)
    
    #cost
    cost = ((h - y) ** 2).sum() / (2*x.size)

    #gradient descent
    derivative_wrt_bias = (h - y).sum() / x.size
    derivative_wrt_weight = np.dot((h - y), x) / x.size
    
    #simultaneous updates
    bias = bias - (learning_rate * derivative_wrt_bias)
    weight = weight - (learning_rate *  derivative_wrt_weight)
    
    return cost


def predict(x):
    global bias, weight
    x = np.array(x)
    
    #x = x / 31 --- feature scaling
    h = bias + (weight * x)
    
    return h
    
def graphCost(cost, bias, weight):
		fig = plt.figure()
		axes = fig.add_subplot(111, projection='3d')
		
		bias = np.array(bias)
		weight = np.array(weight)
		cost= np.array(cost)

		x,y = np.meshgrid(bias, weight)
		z = cost.reshape(x.shape[0],1)
		
		axes.plot_surface(x, y, z, cmap='viridis')
		axes.set_xlabel("bias")
		axes.set_ylabel("weight")
		axes.set_zlabel("cost")

		plt.show()

##############################################################################################
'''
# Overshoot because the learning rate is too high
x = [3,21,31]
y = [3,21,31]

bias = 0.0
weight = 0.0
learning_rate = 0.05
cost = []
b = []
w = []
for i in range(150):
    #print("cost =",fit(x,y),"| bias =", bias, "| weight =", weight)
    cost.append(fit(x,y))
    b.append(bias)
    w.append(weight)
    
plt.plot(list(range(1,151)),cost)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("Overshooting the minimum at learning rate --> 0.05")
predict([1,2,3])
graphCost(cost, b, w)
'''
#############################################################################################

x = [3,21,31]
y = [3,21,31]

bias = 0.0
weight = 0.0
learning_rate = 0.0005
cost = []
b = []
w = []
for i in range(100):
    #print("cost =",fit(x,y),"| bias =", bias, "| weight =", weight)
    cost.append(fit(x,y))
    b.append(bias)
    w.append(weight)
    
plt.plot(list(range(1,101)),cost)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("Lower learning rate --> 0.0005")
predict([1,2,3])
graphCost(cost, b, w)

###################################################
