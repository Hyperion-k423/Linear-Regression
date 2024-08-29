import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('ex1data1.txt',names=['population','profit'])
data.head()
data.describe()
data.info()
data.plot.scatter('population','profit',label='Population')
plt.show()
data.insert(0,'one',1)
data.head()
X=data.iloc[:,0:-1]
X.head()
y=data.iloc[:,-1]
y.head()
X=X.values
X.shape
y=y.values
y.shape
y=y.shape(97,1)
y.shape
def costFunction(X,y,theta):
    inner=np.power(X@theta-y,2)
    return np.sum(inner)/(2*len(X))
theta=np.zeros((2,1))
theta.shape
cost_init=costFunction(X,y,theta)
print(cost_init)
def gradientDescent(X,y,theta,alpha,iters):
    cost=[]
    for i in range(iters):
        theta=theta-(X.T@(X@theta-y)*alpha/len(X))
        cost=costFunction(X,y,theta)
        cost.append(cost)
        if i%100==0:
            print(cost)
    return theta,cost
alpha=0.02
iters=2000
theta,cost=gradientDescent(X,y,theta,alpha,iters)
fig,ax=plt.subplot()
ax.plot(np.arange(iters),cost)
ax.set(xlabel='iters',ylabel='cost',title='cost vs iters')
plt.show()
x=np.linspace(y.min(),y.max(),100)
y=theta[0,0]+theta[1,0]*x
fig,ax=plt.subplots()
ax.scatter(X[:,1],y,label='training data')
ax.plot(x,y,'r',label='predict')
ax.set(xlabel='population',ylabel='profit')
plt.show()






