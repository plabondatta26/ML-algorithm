import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
df=pd.read_csv("Position_Salaries.csv")
x=df.iloc[:,1:2].values
y=df.iloc[:,2].values

#fitting decission  tree regression

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(x,y)


#predicting value

#ypred=reg.predict([[6.5]])
#print(ypred)

#visualizing

# lower resulation
"""plt.scatter(x,y,color='r')
plt.plot(x,reg.predict(x),color='blue')
plt.title("Truth or bluff (Decision Tree)")
plt.xlabel("Position Level")
plt.ylabel("Sallery")
plt.show()"""


#Higher resulation
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='r')
plt.plot(x_grid,reg.predict(x_grid),color='blue')
plt.title("Truth or bluff (Decision Tree: Higher resulation)")
plt.xlabel("Position Level")
plt.ylabel("Sallery")
plt.grid()
plt.show()