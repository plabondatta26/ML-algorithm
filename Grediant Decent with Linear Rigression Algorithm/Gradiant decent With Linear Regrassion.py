import numpy as np
def gredient_descent(x,y):
    m_curr=b_curr=0
    itta=1000
    n=len(x)
    lr=0.001
    for i in range(itta):
        y_predict=m_curr*x+b_curr
        md=-(2/n)*sum(x*(y-y_predict)) #derivative of m
        bd=-(2/n)*sum(y-y_predict) ##derivative of b
        m_curr=m_curr-lr*md
        b_curr=b_curr-lr*bd
        print("m{}, b{}, itta{}".format(m_curr,b_curr,i))

#x=np.array([10,15,13,18])
#y=np.array([20,23,25,18])
#gredient_descent(x,y)

y_pred=m*x + b_curr
print(y_pred)