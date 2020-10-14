import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df= pd.read_csv("Social_Network_Ads.csv")
#print(df.head(10))
x=df.iloc[:,[3]].values
y=df.iloc[:,4].values
#print(df.iloc[:,[3]])
#print(y)

plt.scatter(x,y,marker="+",facecolor="red")
#plt.show()


#spliting data

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


#data preproccesing
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)


#Fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(xtrain,ytrain)
score=classifier.score(xtest,ytest)
print(score)


#predicting the test set result
y_pred=classifier.predict(xtest)
#print("pred output= ",classifier.predict([[26000]]))


#making confusion matris
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
print(cm)

#Visualize the training set data"""








model=LogisticRegression()
len=len(x)
t=int((len/3)*2)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
model.fit(xtrain,ytrain)
y_prede= model.predict(xtest)
print('fff',y_prede)
print(model.score(xtest,ytest), "score 2")

#print("pred-2=",model.predict([[112000]]))
