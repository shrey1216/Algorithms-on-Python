# -*- coding: utf-8 -*-
"""
Problem T-1

https://www.geeksforgeeks.org/monte-carlo-integration-in-python/
"""

from typing import no_type_check
from scipy import random
import numpy as np
import math

number = 0
for i in range(4000):
  

  # Upper and Lower bounds for integration. Going from -Rad2 to Rad2
  a = -1 * math.sqrt(2)
  b = math.sqrt(2)
  N = 1000

  # array of zeros of length N
  ar = np.zeros(N)

  # fill ar with random values within the interval [a,b]
  for i in range (len(ar)):
    ar[i] = random.uniform(a,b)

  # Store functions of the variable x. Starts at 0
  integral = 0.0
  discIntegral = 0.0

  # Function of the heart as a method 
  def fHeart(x):
    return math.sqrt(2-(x**2)) + math.sqrt(abs(x))

  def fDisc(x):
    return math.sqrt(2-(x**2)) + 1.35


  # iterates and sums up values of different functions
  # of x
  for i in ar:
    integral += fHeart(i)
    discIntegral += fDisc(i)

  # we get the answer by the formula derived adobe
  ans = (b-a)/float(N)*(integral + discIntegral)


  number += ans

#print solution
print ("The value calculated as the overlapping of both figures is {}.".format(number/4000))

"""Problem T-2

https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/
"""

def bisectionMethod(function,a,b,N):

#intial check can't be higher or equal to 0
    if function(a)*function(b) >= 0:
        print("No point found (check failed")
        return None


    a1 = a
    b1 = b

    #new range divide by 2 
    #N iterations which is inputted 
    for n in range(1,N+1):
        #newRange = c
        newRange = (a1 + b1)/2
        fNew = function(newRange)
        
        #f(a)f(c) < 0, then b = c, else a = c
        if function(a1)*fNew < 0:
            a1 = a1
            b1 = newRange
        #a = c
        elif function(b1)*fNew < 0:
            a1 = newRange
            b1 = b1

        #if f(c) = 0 then c is the exact root solution
        elif fNew == 0:
            print("Found exact solution.")
            return newRange
            #after checking both sides not found
        else:
            print("No point found")
            return None

    #return the final range / 2 as the root after N iterations have passed 
    return (a1 + b1)/2

function = lambda x: (2.020**((-x)**3))-((x**3)*math.cos(x**4))-1.984

#guess 1: -0.8
#range: [-0.9, -0.7]
x1 = bisectionMethod(function,-0.9,-0.7,1000)

#guess 2: 1.25
#range: [1.20, 1.30]
x2 = bisectionMethod(function,1.2,1.3,1000)
#guess 3: 1.4
#range: [1.35, 1.45]
x3 = bisectionMethod(function,1.35,1.45,1000)
#guess 4: 1.7
#range: [1.65, 1.75]
x4 = bisectionMethod(function,1.65,1.75,1000)
#guess 5: 1.8
#range: [1.75,1.85]
x5 = bisectionMethod(function,1.75,1.85,1000)
#guess 6: 1.95
#range: [1.9, 2]
x6 = bisectionMethod(function,1.9,2,1000)

print("The first root is", x1)
print("The second root is", x2)
print("The third root is", x3)
print("The fourth root is", x4)
print("The fifth root is", x5)
print("The sixth root is", x6)
