import pandas as  pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from math import sqrt

#loading dataset
df=pd.read_csv('Social_Network_Ads.csv')
#print(df)
x= df.iloc[:,[2,3]]
y= df.iloc[:,4]


#splitng data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Preprocessing
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#print(x_test)
n=int(sqrt(len(y_train)))
print(n)
#fitting classifire into training set
classifire=KNeighborsClassifier(n_neighbors=n,metric='minkowski',p=2)
classifire.fit(x_train,y_train)

#Predicting test set result
y_pred=classifire.predict(x_test)
#print(y_pred)

#making confusion metrix
cm=confusion_matrix(y_test,y_pred)
#print(cm)

#plotting training set
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                   np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1,x2,classifire.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('white','blue')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
