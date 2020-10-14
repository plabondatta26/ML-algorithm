import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Salary_Data.csv')
X=df.iloc[:, :-1].values.reshape(-1, 1)
Y=df.iloc[:, 1].values.reshape(-1, 1)

#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#print(len(X))

training_X,training_Y=X[:20],Y[:20]
test_X,test_Y=X[21:],Y[21:]
reg=LinearRegression()
reg.fit(training_X,training_Y)
regline=reg.predict(X)
plt.scatter(X,Y,color='r')
plt.plot(X,regline,marker='o')
y_pred=reg.predict(test_X)
plt.grid()
#print(y_pred)
plt.show()
