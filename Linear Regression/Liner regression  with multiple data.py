#importing libreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
#importing dataset
data=pd.read_csv("Weather.csv")
#print(data)
#print (data.columns)
reg=LinearRegression()
reg.fit(data[['Month','Max Temp','Min Temp']],data.ColdCoverage)
reg.coef_
reg.intercept_
print('m=',reg.coef_, 'c=',reg.intercept_)
print('may be the cold coverage will be : ')
print('y=mx+c So, y=',reg.predict([[11,36.12,18.00]]))