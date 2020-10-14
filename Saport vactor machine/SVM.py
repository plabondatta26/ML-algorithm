from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

#loading data
cancer_data=datasets.load_breast_cancer()
print(cancer_data)

#spliting data
x_tarin,x_test,y_train,y_test=train_test_split(cancer_data.data,cancer_data.target,test_size=0.4,random_state=209)

#seperable style (linear/ non-linear)
cls=svm.SVC(kernel='linear')


#train model
cls.fit(x_tarin,y_train)
print('score:',cls.score(x_tarin,y_train))
#predicting
pred=cls.predict(x_test)
#print("Accuracy=",round(metrics.accuracy_score(y_test,y_pred=pred),2))

#precision score
#print("Precision=",round(metrics.precision_score(y_test,y_pred=pred),2))

#recall score
#print("Recall=",round(metrics.recall_score(y_test,y_pred=pred),2))

#Classification Report
#print("Classification Report=",metrics.classification_report(y_test,y_pred=pred))