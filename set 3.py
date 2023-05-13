# -*- coding: utf-8 -*-
"""
Problem 3.1

Sources:

https://stackoverflow.com/questions/38772640/rebinning-a-list-of-numbers-in-python

https://towardsdatascience.com/how-to-easily-create-tables-in-python-2eaea447d8fd

https://python4mpia.github.io/fitting_data/least-squares-fitting.html

https://www.codesansar.com/numerical-methods/runge-kutta-fourth-order-rk4-python-program.htm


(1) Generate 10^4 numbers uniformly distributed within (-2,2)
"""

import numpy as np
import scipy.optimize as optimization

N = 10**4
u = np.random.uniform(-2, 2, N)
print(u)

"""(2) Generate M = 10^3 numbers by adding groups of ten of ui"""

#M = 10**3
x = []
count = 0
temp = 0
for a in range(len(u)):
  count += 1
  temp += u[a]
  if(count == 10):
    x.append(temp)
    count = 0
    temp = 0

print(x)

"""(3) Bin the numbers in x into 8 boxes and tabulate the results """

#import pandas as pd
#from prettytable import PrettyTable
from tabulate import tabulate

#a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

bins=[-20,-15,-10,-5,0,5,10,15,20]
hist, bin_edges = np.histogram(x, bins=bins)

info = {'Bin Interval': ['(-20,-15)', '(-15,-10)', '(-10,-5)', '(-5,0)', '(0,5)', '(5,10)', '(10,15)', '(15,20)'], 'Bins g': ['-4', '-3', '-2', '-1', '1', '2', '3', '4'], 'Bin Counts P(g)': hist}
#print(tabulate(info, headers='keys'))
print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

"""(4) Fit the last two columns into the given form"""

xData = np.array([-4,-3,-2,-1,1,2,3,4])
yData = np.array(hist)
x0    = np.array([0.0, 0.0, 0.0]) 
sigma = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])


#g is x
#P(g) is y
#def func(x, a, b, c):
#    return a + b*x + c*x*x
xData = np.transpose(np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
              [-4,-3,-2,-1,1,2,3,4]]))

#def func(params, xData, yData):
##    return (yData - np.dot(xData, params))

#print(optimization.curve_fit(func, xData, yData, x0, sigma))

#print(optimization.leastsq(func, x0, args=(xData, yData)))

"""(5) Plot the values from (4) with our newly found alpha and beta values over [-4.4]

Problem 3.2
(1) Apply iteration method to find trajectory directly

Euler method, Midpoint Method, Runge-Kutta methods
"""

# RK-4 method python program

# function to be solved
def f(x,y):
    return ((y / x) - (1 / 3) * (1 + (y / x)**2)**(1 / 2))

# RK-4 method
def rk4(x0,y0,xn,n):
    
    # Calculating step size
    h = (xn-x0)/n
    for i in range(n):
        k1 = h * (f(x0, y0))
        k2 = h * (f((x0+h/2), (y0+k1/2)))
        k3 = h * (f((x0+h/2), (y0+k2/2)))
        k4 = h * (f((x0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        yn = y0 + k

        y0 = yn
        x0 = x0+h   
    return yn

#Inputs
#Initial Condition
#a = 100
x0 = 100

#y(100) = 0
y0 = 0

#Calculation point
xn = 1

#print('Enter number of steps:')
step = 2

# RK4 method call
#rk4(x0,y0,xn,step)

hundred = []
for x in range(100):
  hundred.append(x+1)

values = []

for x in range(100):
  values.append(rk4(x0,y0,x+1,step))


print("Trajectory of the plane: ")
info = {'X Value': hundred, 'Y Value': values}
print(tabulate(info, headers='keys', tablefmt='fancy_grid'))
