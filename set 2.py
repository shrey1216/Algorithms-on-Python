# -*- coding: utf-8 -*-
"""
2.1) Matrix Multiplication (naiv algorithm)

https://www.programiz.com/python-programming/matrix

https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/

https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html

https://www.geeksforgeeks.org/multiplication-two-matrices-single-line-using-numpy-python/

https://www.interviewbit.com/blog/strassens-matrix-multiplication/

(1) Generate two matrices of dimension 2^10 by 2^10
"""

import numpy as np
from scipy.linalg import solve

rows, cols = (2**10, 2**10)

arr1 = [[np.random.uniform(-1,1,None) for i in range(cols)] for j in range(rows)]
arr2 = [[np.random.uniform(-1,1,None) for i in range(cols)] for j in range(rows)]

"""(2) Matrix multiplication using naive algorithm (and add number of additions and multiplications)."""

arr3 = [[0 for i in range(cols)] for j in range(rows)]

#multiplications = 0
for i in range(len(arr1)):
    for j in range(len(arr2[0])):
        for k in range(len(arr2)):
 
            #resulted matrix
            arr3[i][j] += arr1[i][k] * arr2[k][j]
            #multiplications += 1


#additions = multiplications / 2
#print(additions)
#print(multiplications)


#ESTIMATE THE NUMBER FOR 2**10
#this number is for the number of additions
#the number of multiplications would be double (2 for every 1 addition)




for row in arr3:
    print(row)

"""(3) Strassen Algorithm (estimate number of additions and multiplications)"""

def strassen(x, y):
    if x.size == 1 or y.size == 1:
        return x * y

    n = x.shape[0]

    if n % 2 == 1:
        x = np.pad(x, (0, 1), mode='constant')
        y = np.pad(y, (0, 1), mode='constant')

    m = int(np.ceil(n / 2))
    a = x[: m, : m]
    b = x[: m, m:]
    c = x[m:, : m]
    d = x[m:, m:]
    e = y[: m, : m]
    f = y[: m, m:]
    g = y[m:, : m]
    h = y[m:, m:]
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)
    result = np.zeros((2 * m, 2 * m), dtype=np.int32)
    result[: m, : m] = p5 + p4 - p2 + p6
    result[: m, m:] = p1 + p2
    result[m:, : m] = p3 + p4
    result[m:, m:] = p1 + p5 - p3 - p7

    return result[: n, : n]


x = np.array(arr1)
y = np.array(arr2)

print(x)
print('')
print(y)
print('')

print(strassen(x, y))

"""2.2) Solve the system of linear equations (jacobi, GS, and SOR methods w = 1.1)

https://www.epythonguru.com/2020/10/jacobi-method-in-python-and-numpy.html

https://www.geeksforgeeks.org/gauss-seidel-method/

https://stackoverflow.com/questions/53251299/successive-over-relaxation

Jacobi Method
"""

def jacobi(A,b,x,n):
  D = np.diag(A)
  R = A-np.diagflat(D)
  for i in range(n):
    x = (b-np.dot(R,x))/D
  return x

# first matrix
A = np.array([[7,-1,-6,0],[-5,-4,10,8],[0,9,4,-2],[1,0,-7,10]])

# second matrix 
b = [11,21,-12,-1]

# guess
x = [1.0,1.0,1.0,1.0]

# iterations
n = 25

# x = algorithm solution
x = jacobi(A,b,x,n)

print(solve(A,b))

"""Gauss-Seidel Method"""

def seidel(a, x ,b):
	#Finding length of a(4)	
	n = len(a)				
	# for loop for 4 times since n = 4
	for j in range(0, n):		
		
		d = b[j]				
		
		
		for i in range(0, n):	
			if(j != i):
				d -= a[j][i] * x[i]
				
		x[j] = d / a[j][j]
	#return value of solution		
	return x	

# this is the number of variables in the system 				
n = 4	

a = []							
b = []		
# 4 zeros for n = 4					
x = [0, 0, 0, 0]	
			
a = [[7,-1,-6,0],[-5,-4,10,8],[0,9,4,-2],[1,0,-7,10]]

b = [11,21,-12,-1]

#loop until the correct solution is found.
for i in range(0, 25):			
	x = seidel(a, x, b)

print(x)

"""Successive Over-Relaxation Method (w = 1.1)"""

def SOR(A, b, omega, guess, convergence):
  phi = guess[:]
  residual = np.linalg.norm(np.matmul(A, phi) - b) #Initial residual
  while residual > convergence:
    for i in range(A.shape[0]):
      sigma = 0
      for j in range(A.shape[1]):
        if j != i:
          sigma += A[i][j] * phi[j]
      phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)
    residual = np.linalg.norm(np.matmul(A, phi) - b)
    print('Residual: {0:10.6g}'.format(residual))
  return phi

residual = 1e-8
omega = 1.1 
#omega = 0.5 

A = np.ones((4, 4))
A[0][0] = 7
A[0][1] = -1
A[0][2] = -6
A[0][3] = 0

A[1][0] = -5
A[1][1] = -4
A[1][2] = 10
A[1][3] = 8

A[2][0] = 0
A[2][1] = 9
A[2][2] = 4
A[2][3] = -2

A[3][0] = 1
A[3][1] = 0
A[3][2] = -7
A[3][3] = 10

b = np.ones(4)
b[0] = 11
b[1] = 21
b[2] = -12
b[3] = -1

guess = np.zeros(4)

x = SOR(A, b, omega, guess, residual)
print(x)
