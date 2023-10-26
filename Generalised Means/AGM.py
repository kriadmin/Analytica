from math import prod
from functools import reduce

import numpy as np


def AM(*args):
    return sum(args)/len(args)
def GM(*args):
    return prod(args) ** (1./len(args))


def AGM(t): 
    return (GM(*t),AM(*t))

def AGM_repeated(t,n):
    return reduce(lambda x,y : AGM(x),range(n),t)


""" import plotly.graph_objects as go

A = np.linspace(1,200,400)
B = np.linspace(1,200,400)

X,Y = np.meshgrid(A,B)

Z = AGM_repeated((X,Y),10)

fig = go.Figure(data = [go.Surface(z=Z[0],x=X,y=Y)])
fig.show()
 """

import matplotlib.pyplot as plt

A = np.linspace(0.000000000001,10,800000)
R = AGM_repeated((100,A),5)

a,b,c,d = np.polyfit(A**0.5, R[0], 3)

print(a,b,c,d)

Y = a*A**1.5 + b*A + c*A**0.5 + d
err = np.square(Y - R[0]).mean()
print(err)
plt.plot(A,R[0])
#plt.plot(A,Y)
plt.show()
