#importing libreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
#importing dataset
dataset = pd.read_csv('Weather.csv')
#print (dataset.columns)

X=dataset["Max Temp"].values.reshape(-1,1)
Y=dataset["Min Temp"].values.reshape(-1,1)
#print(len(X))


#Set up data's to test and training set

training_x,training_y=X[0:14080],Y[0:14080]
test_x,test_y=X[14081,:],Y[14081,:]

#getting the value of M &C
reg=LinearRegression()
reg.fit(X,Y)
regLine=reg.predict(X)
plt.scatter(X,Y,color='red',marker="o")
plt.plot(X,regLine)
plt.xlabel("Max Tempareture")
plt.ylabel("Minimum Tempareture")
#plt.show()

#print(reg.score(X,Y))

y=(reg.coef_*43)+reg.intercept_
print("m =",reg.coef_)
print("c =",reg.intercept_)
print("y =",y)