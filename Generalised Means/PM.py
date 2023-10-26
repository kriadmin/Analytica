from math import prod
from functools import reduce

import numpy as np


def AM(*args):
    return sum(args)/len(args)
def GM(*args):
    return prod(args) ** (1./len(args))
def HM(*args):
    return len(args) / sum(1. / val for val in args)



def PM(t): 
    return (HM(*t),GM(*t),AM(*t))

def PM_repeated(t,n):
    return reduce(lambda x,y : PM(x),range(n),t)

""" 

import plotly.graph_objects as go

A = np.linspace(1,200,400)
B = np.linspace(1,200,400)

X,Y = np.meshgrid(A,B)
Z = np.full(X.shape,2.0)

R = PM_repeated((X,Y,Z),10)

fig = go.Figure(data = [go.Surface(z=R[0],x=X,y=Y)])
fig.show()

"""
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

A = np.linspace(1,40000,8000)
R = PM_repeated((10,10,A),5)

a,b,c = np.polyfit(A**0.5, R[0], 2)

Y = a*A + b*A**0.5 + c

print(a,b,c)
err = np.square(Y - R[0]).mean()
print(err)

plt.plot(A,R[0])
plt.plot(A,Y)
plt.show()

#print(R[0].shape)
# print(PM_repeated(s,0))
# print(PM_repeated(s,1))
#print(PM_repeated(s,20))