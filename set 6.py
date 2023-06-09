# -*- coding: utf-8 -*-
"""

(1) Using the following parametric curve, find the area of the space enclosed within the red curve. 

Monte Carlo method:
"""

#plot parameterized function
#x = t**5 - 4*t**3
#y = t**2

ptsIn = 0
ptsTotal = 10000000
for i in range(ptsTotal):
  randx = random.random()*3 - 1.5
  randy = random.random()*4 - 1.5
  if randx**2 + (randy - np.sqrt(abs(randx)))**2 <= 2:
    ptsIn += 1
parameterizedFunction = (ptsIn/ptsTotal)*(3*4)

"""Problem F-3

(1) Table of values for the next 250 days

Source: https://www.tutorialspoint.com/python_data_science/python_normal_distribution.htm

HW3 Suggested Solutions
"""

import numpy as np
from random import *
from tabulate import tabulate
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt


#np.random.normal asks for the mu and sigma (standard deviation, not the variance)
mu1, sigma1 = -0.005, 0.015 # mean and standard deviation
mu2, sigma2 = 0.010, 0.013

#x is the dow index's change rate 
#Dow(t+1) = Dow(t)(1 + x)


#make empty array for dow for the first 60 days 
dow = []

#we only need 59 values now
for x in range(60):
  #if x equals 0, then a x-1 value does not exist. 
  if(x == 0):
    #add original value
    s = np.random.normal(mu1, sigma1,1)
    dow.append(1 + s[0])
  else:
    s = np.random.normal(mu1, sigma1,1)
    #add a new value (previous value times 1 + new change rate)
    dow.append(dow[x-1] * (1 + s[0]))

#the next 190 days
for x in range(190):
    #if x equals 0, then a x-1 value does not exist. 
  if(x == 0):
    s = np.random.normal(mu2, sigma2,1)
    dow.append(1 + s[0])
  else:
    s = np.random.normal(mu2, sigma2,1)
    #add a new value (previous value times 1 + new change rate)
    dow.append(dow[x-1] * (1 + s[0]))

days = []
for x in range(250):
  days.append(x+1)

info = {'Day': days, 'Dow index value': dow}
#print(tabulate(info, headers='keys'))
print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

"""(2) Six even data points from the first 60 days (5th, 15th, 25th, 35th, 45th, 55th) and fit them into the function """

def func(x, D, b):
  return D*np.exp(b*x)

dayVals = [4,14,24,34,44,54]
dowVals = [dow[4],dow[14],dow[24],dow[34],dow[44],dow[54]]
param_vals, _ = curve_fit(func, dayVals, dowVals)

a2, b2 = param_vals
print("alpha is", a2) 
print("beta is", b2)

"""Problem F-4

IVP Problem
"""

a = 8
v0 = 20
vb = 30


def dydx(x, y, b):
  return ((v0 / vb) * (1 - ((x**2)/(a**2))) * (1 / math.cos(b)) + (math.sin(b) / math.cos(b)))

#forwardEuler Method
def forwardEuler(x0, y0, x, h,b):
  n = (int)((x - x0)/h)
  y = y0
  result = [y]
  for i in range(n):
    y = y + h * dydx(x0,y,b)
    x0 += h
    result.append(y)
  return y, result

x0 = a
y0 = 0
x = 0
n = 10000
h = (x - x0)/n

results = []
for x in range(360):
  y_Euler, y_line_Euler = forwardEuler(x0, y0, x, h,x)
  results.append(abs(y_Euler))

print('The values for each angle value (absolute value), [0-360), are shown: ' + str(results))
print('The minimum value is: ' + str(min(results)))
print('The angle at this value is: ' + str(results.index(min(results))) + ' degrees which is the optimal value at which the boat travels on the shortest path')
