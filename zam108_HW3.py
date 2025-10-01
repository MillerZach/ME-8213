import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import numpy as np

def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b

def curvature(xData,yData):
    n = len(xData) - 1
    c = np.zeros(n)
    d = np.ones(n+1)
    e = np.zeros(n)
    k = np.zeros(n+1)
    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0 * (xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] = 6.0 * (yData[0:n-1] - yData[1:n]) \
        /(xData[0:n-1] - xData[1:n]) \
        - 6.0 * (yData[1:n] - yData[2:n+1]) \
        /(xData[1:n] - xData[2:n+1])
    LUdecomp3(c,d,e)
    LUsolve3(c,d,e,k)
    return k

def slope(xData,yData):
    n = len(xData) - 1
    c = np.zeros(n)
    d = np.ones(n+1)
    e = np.zeros(n)
    k = np.zeros(n+1)
    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0 * (xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] = ((xData[0] - xData[1])/3) \
        + (xData[0] - xData[1])/6 \
        + (yData[0] - yData[1]) \
        /(xData[0] - xData[1])
    LUdecomp3(c,d,e)
    LUsolve3(c,d,e,k)
    return k

def evalSpline(xData,yData,k,x):

    def findSegment(xData,x):
        iLeft = 0
        iRight = len(xData) - 1
        while 1:
            if (iRight-iLeft) <= 1: return iLeft
            i = int((iLeft + iRight)/2)
            if x < xData[i]:
                iRight = i
            else:
                iLeft = i

    i = findSegment(xData,x)
    h = xData[i] - xData[i+1]
    y = ((x - xData[i+1])**3/h - (x - xData[i+1])*h)*k[i]/6.0 \
      - ((x - xData[i])**3/h - (x - xData[i])*h)*k[i+1]/6.0     \
      + (yData[i]*(x - xData[i+1])                            \
      - yData[i+1]*(x - xData[i]))/h

    return y

'''
data = np.array([[0.0, 1.225], [3.0, 0.905], [6.0, 0.652]])


quadratic = np.poly1d(np.polyfit(data[:,0], data[:,1], 2))
quadratic_fit = quadratic([data[:,0]])

plt.scatter(data[:,0], data[:,1], marker='o', color='red', label='Actual')
plt.scatter(data[:,0], quadratic_fit, marker='.', color='blue', label='Quadratic')

plt.grid(True)
plt.ylabel('Height (km)')
plt.xlabel('Density (kg/m^3)')
plt.legend()

plt.show()

four_k = quadratic([4.0])

print("Density at 4km : ")
print(four_k)

h_max_range = np.linspace(0.0, 80.0, 40)
density_range = quadratic([h_max_range])
min_density =  np.min(density_range)
min_alt = np.argmin(density_range)
print("\nLowest density: ")
print(min_density)

print("At altitude: ")
print(h_max_range[min_alt])'''

xData = np.array([0.0, 1.0, 2.0, 3.0])
yData = np.array([1.0, 1.0, 0.5, 0.0])

k = curvature(xData,yData)
print(k)

x = eval(input("\nx ==> "))
print("y =",evalSpline(xData,yData,k,x))

k = slope(xData,yData)
print(k)
x = eval(input("\nx ==> "))
print("y =",evalSpline(xData,yData,k,x))

# Check using SciPy CubicSpline Function
spline = CubicSpline(xData,yData, bc_type='natural')
prediction = spline(2.6)
print(prediction)

spline = CubicSpline(xData,yData, bc_type='clamped')
prediction = spline(2.6)
print(prediction)

'''x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
y = np.array([0.49, 1.6, 3.36, 6.44, 10.16])

plt.scatter(x, y, marker='o', color='red')
plt.show()'''