from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
df=datasets.load_digits()
#print(df)
cls=svm.SVC(gamma=0.001,C=100)
x,y=df.data[:-10],df.target[:-10]
cls.fit(x,y)

print(cls.predict(df.data[:-10]))

plt.imshow(df.images[10],interpolation='nearest')
plt.show()