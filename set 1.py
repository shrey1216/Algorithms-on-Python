# -*- coding: utf-8 -*-
"""
Problem 1.2

References: 

1. Making the graph: https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/

2. Bisection method in python: https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/

3. Newton's method in python: https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/


(1)
"""

import matplotlib.pyplot as plt
import numpy as np
  
# x coordinate from -1 to 4
x = np.arange(-1, 4, 0.05)
# y coordinate using function given in question
y = (x**5) - (5 * (x**4)) + (5 * (x**3)) + (5 * (x**2)) - (6 * x) - 1

# plotting points on graph
plt.plot(x, y)
# print plot
plt.show()

"""(2) From the graph, I can eyeball one of the roots to be x=0. I will label this x0. 

(3) The interval chosen so that there is only one root contained within it is [-0.5,0.5] where the value of δ is 0.5. 

(4)
"""

def bisectionMethod(function,a,b,N):

#intial check can't be higher or equal to 0
    if function(a)*function(b) >= 0:
        print("No point found")
        return None
    a1 = a
    b1 = b

    #new range divide by 2 
    #N+1 iterations 
    for n in range(1,N+1):
        newRange = (a1 + b1)/2
        fNew = function(newRange)
 
        if function(a1)*fNew < 0:
            a1 = a1
            b1 = newRange
        elif function(b1)*fNew < 0:
            a1 = newRange
            b1 = b1

        elif fNew == 0:
            print("Found exact solution.")
            return newRange
            #after checking both sides not found
        else:
            print("No point found")
            return None
    return (a1 + b1)/2

function = lambda x: (x**5) - (5 * (x**4)) + (5 * (x**3)) + (5 * (x**2)) - (6 * x) - 1
bisectionMethod(function,-0.5,0.5,100)

"""x1 is -0.1509841732749716 or -0.151 (4 digits of accuracy)

(5) From the graph, another root I can guess is x=3. I will label this x2. I will now use Newton's method to perfect this root. 
"""

def newton(function,DerF,guess,ep,max):
    xn = guess
    for n in range(0,max):
        functionNew = function(xn)
        if abs(functionNew) < ep:
            print('Found solution:',n,'iterations.')
            return xn
        Dfxn = DerF(xn)
        if Dfxn == 0:
            print('No solution')
            return None
        xn = xn - functionNew/Dfxn
    print('No solution found (max reached)')
    return None

function = lambda x: (x**5) - (5 * (x**4)) + (5 * (x**3)) + (5 * (x**2)) - (6 * x) - 1
derFunction = lambda x: (5*(x**4)) - (20 * (x**3)) + (15 * (x**2)) + (10 * x) - 6 
approx = newton(function,derFunction,3,1e-13,5)
print(approx)

"""x2 is 3.038495291036736 or 3.0385 (4 digits of accuracy)

Problem 1.3

References: 

1. Simplify equations in python: https://www.geeksforgeeks.org/python-sympy-simplify-method/


(1)
"""

x1 = 10
y1 = 14269.53

x2 = 20
y2 = 13830.47

x3 = 30
y3 = 13647.43

x4 = 40
y4 = 13168.80

x5 = 50
y5 = 12710.42

x = [x1,x2,x3,x4,x5]
y = [y1,y2,y3,y4,y5]

plt.scatter(x, y)
plt.plot(x, y)
plt.title("NASDAQ data for the year 2022")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

from sympy import simplify 

x = symbols('x')

expr = ((y1 * (x - x2) * (x - x3) * (x - x4) * (x - x5)) / ((x1 - x2) * (x1 - x3) * (x1 - x4) * (x1 - x5)) + (y2 * (x - x1) * (x - x3) * (x - x4) * (x - x5)) / ((x2 - x1) * (x2 - x3) * (x2 - x4) * (x2 - x5)) + (y3 * (x - x1) * (x - x2) * (x - x4) * (x - x5)) / ((x3 - x1) * (x3 - x2) * (x3 - x4) * (x3 - x5)) + (y4 * (x - x1) * (x - x2) * (x - x3) * (x - x5)) / ((x4 - x1) * (x4 - x2) * (x4 - x3) * (x4 - x5)) + (y5 * (x - x1) * (x - x2) * (x - x3) * (x - x4)) / ((x5 - x1) * (x5 - x2) * (x5 - x3) * (x5 - x4)))
   
# Use sympy.simplify() method
smpl = simplify(expr) 
    
print("y = {}".format(smpl))

"""Using the Lagrange interpolation polynomial for the data points, the polynomial y = 0.00361437500000005*x^4 - 0.453372500000002*x^3 + 19.4465125*x^2 - 
364.15625*x + 16383.67 is found.

(2) Interpolate sin(2x) at [0, pi/4] using the Chebyshev method

Error upper bound
"""

a = 0
b = math.pi / 4
n = 4
upperBound = (((b - a) / 2) ** n) / (math.factorial(4) * 2**(n-1))

print(upperBound)
