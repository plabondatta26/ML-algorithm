# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Pijja.csv')

#print(dataset.columns)

X=dataset["Size"].values.reshape(-1,1)
Y=dataset["Price"].values.reshape(-1,1)

#Splitting  dataset to testing and training

training_x,training_y=X[0:3],Y[0:3]
test_x,test_y=X[4:],Y[4:]
model=LinearRegression()
model.fit(X,Y)
regretion_line=model.predict(X)
plt.plot(X,regretion_line)
plt.plot(training_x,training_y,'o')
plt.plot(test_x,test_y,'g')

plt.scatter(X,Y,marker="v")
plt.scatter(X,Y,marker=".")
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Pizza Price Prediction')
plt.show()

print("m =",model.coef_)
print("c =",model.intercept_)
#pr=model.predict([[36]])
#print("r =",pr)
y=(107.21491228*36)+(-174.69298246)     # y=mx+c
print("y =",y)