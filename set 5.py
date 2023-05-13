# -*- coding: utf-8 -*-
"""
1) Generate a matrix of 32x32
"""

import numpy as np
from scipy.linalg import solve

rows, cols = (2**5, 2**5)

A = [[np.random.uniform(0,2,None) for i in range(cols)] for j in range(rows)]
print('Matrix A:')
for row in A:
    print(row)

"""(1) Find the product of AB where B = A^T

Source: https://www.geeksforgeeks.org/transpose-matrix-single-line-python/

"""

B = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
print('Matrix B:')
for row in B:
    print(row)


AB = [[0 for i in range(cols)] for j in range(rows)]
for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
 
            #resulted matrix
            AB[i][j] += A[i][k] * B[k][j]
print('')
print('Matrix AB:')
for row in AB:
    print(row)

"""(2) The Power Method

Source: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html

"""

def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n

#32 columns
x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
 
for i in range(8):
    x = np.dot(A, x)
    lambda1, x = normalize(x)
    
print('Eigenvalue:', lambda1)
print('Eigenvector:', x)

"""
2)

"""

import matplotlib.pyplot as plt

#Uniform Mesh Size of 10000
h1 = 10**4


# t value arrays spaced out according to h
# the x will explode later
x1 = np.arange(0, 1, h1)

# list for the DEs
y1method1 = [1]
y1method2 = [1]

def f(c, x, y): #this is the function f(x)
  return (-c + x + y + (2 * x * y))




def bisectionMethod (A, B, x, y):
  a = A
  b = B
  c = (a + b)/2 #defintion of xn according to the bisection method
  if abs(f(c, x, y)) <= 0.00005: #checks the the error is less than
    return c #returns the x value
  if (f(a, x, y) * f(c, x, y)) >= 0:
    return bisectionMethod(c, b, x, y) #eliminates the left side of x using recursion
  if (f(b, x, y) * f(c, x, y)) >= 0:
    return bisectionMethod(a, c, x, y) #eliminates the right side of x using recursion


# h1, foward euler
for i in range(0, len(x1)-1):
  y1method1.append(y1method1[i] + h1 * bisectionMethod(-10000000, 10000000,x1[i], y1method1[i]))
  #print(y1method1[i])


#h1, RK4
for i in range(0, len(x1)-1):
  k1 = bisectionMethod(-10000, 10000, x1[i], y1method2[i])
  k2 = bisectionMethod(-10000, 10000, x1[i] + 1/2*h1, y1method2[i] + 1/2*h1*k1)
  k3 = bisectionMethod(-10000, 10000, x1[i] + 1/2*h1, y1method2[i] + 1/2*h1*k2)
  k4 = bisectionMethod(-10000, 10000, x1[i] + h1, y1method2[i] + h1*k3)
  y1method2.append(y1method2[i] + 1/6*h1*(k1 + 2*k2 + 2*k3 + k4))

# Plot the DE

plt.plot(x1, y1method1, label = '10^5, euler')
plt.plot(x1, y1method2, label = '10^5, RK4')
plt.legend()
plt.show()
