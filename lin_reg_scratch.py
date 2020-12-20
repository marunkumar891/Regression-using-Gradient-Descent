import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")
#print(data.head())

X,y = data.iloc[:, 0],data.iloc[:, 1]
print(X,y)

#data visualisation

plt.scatter(X,y)
plt.show()

#inital declaration

m = 0
c = 0
l = 0.0001         #learning rate - this value should me minimum for a minimization problem
no_of_iter = 1000 #number of iteration required for reaching the global minimum point for optimizing the cost function
n = float(len(X))

for i in range(no_of_iter):
	y_pred = m*X + c 
	#print(y_pred)
	d_m = (-2/n)*sum(X*(y - y_pred)) #partial derivation wrt slope
	d_c = (-2/n)*sum(y - y_pred)     #partial derivation wrt y-intercept
	m = m - l*d_m
	c = c - l*d_c

print(round(m,2),round(c,2)) #this is the required slope and intercept of the regression line

#---------------------------plottting regression line----------------------------------------------------

y_pred = m*X + c
plt.scatter(X, y)
plt.scatter(X, y_pred)
plt.show()

print("the required regression line:","y = ",round(m,2),"X + ",round(c,2))