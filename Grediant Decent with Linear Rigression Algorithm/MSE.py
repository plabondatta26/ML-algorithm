import numpy as np
y=np.array([20,23,25,28])
ypred=np.array([18,25,28,34])
n=len(y)
for i in range(n):
    x=(1/n)*sum((y-ypred)**2) #law=( 1/n)*(y(i)-y_predict(i)). bar bar i er man change er jonno protibar e square kore tarpor jog kore tader mean ber korte hobe.
print(x)