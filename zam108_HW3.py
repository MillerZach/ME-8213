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

def plotPoly(xData,yData,coeff,xlab='x',ylab='y'):
    m = len(coeff)
    x1 = min(xData)
    x2 = max(xData)
    dx = (x2-x1)/20.0
    x = np.arange(x1,x2 + dx/10.0,dx)
    y = np.zeros((len(x)))*1.0
    for i in range(m):
        y = y + coeff[i]*x**i
    plt.plot(xData,yData,'-',x,y,'-', color = 'green')
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid (True)
    plt.show()

def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]

def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))

    for k in range(0,n-1):
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)

        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

def polyFit(xData,yData,m):
    a = np.zeros((m+1,m+1))
    b = np.zeros(m+1)
    s = np.zeros(2*m+1)
    for i in range(len(xData)):
        temp = yData[i]
        for j in range(m+1):
            b[j] = b[j] + temp
            temp = temp*xData[i]
        temp = 1.0
        for j in range(2*m+1):
            s[j] = s[j] + temp
            temp = temp*xData[i]
    for i in range(m+1):
        for j in range(m+1):
            a[i,j] = s[i+j]
    return gaussPivot(a,b)

def stdDev(c,xData,yData):
    def evalPoly(c,x):
        m = len(c) - 1
        p = c[m]
        for j in range(m):
            p = p*x + c[m-j-1]
        return p

    n = len(xData) - 1
    m = len(c) -1
    sigma = 0.0
    for i in range(n+1):
        p = evalPoly(c,xData[i])
        sigma = sigma + (yData[i] - p)**2
    sigma = math.sqrt(sigma/(n-m))
    return sigma

# Problem 1

# Manually defining data points
data = np.array([[0.0, 1.225], [3.0, 0.905], [6.0, 0.652]])

# Quadratic based on density and height
quadratic = np.poly1d(np.polyfit(data[:,0], data[:,1], 2))
quadratic_fit = quadratic([data[:,0]])

# Plot actual data vs quadratic prediction within recorded data's range
plt.scatter(data[:,0], data[:,1], marker='o', color='red', label='Actual')
plt.scatter(data[:,0], quadratic_fit, marker='.', color='blue', label='Quadratic')

plt.grid(True)
plt.ylabel('Height (km)')
plt.xlabel('Density (kg/m^3)')
plt.legend()

plt.show()

# using quadratic fit to predict density at 4km
four_k = quadratic([4.0])

print("Density at 4km : ")
print(four_k)

# Defining array for h that goes up to the highest altitude for earth's atmosphere
h_max_range = np.linspace(0.0, 80.0, 40)

# predicting density within that range
density_range = quadratic([h_max_range])

# Finding lowest density value
min_density =  np.min(density_range)

# Using index of lowest density to find its height
min_alt = np.argmin(density_range)

# plotting to check minimum

plt.plot(density_range[0,:], h_max_range, 'b-', label='Density')
plt.xlabel('Density (kg/m^3)')
plt.ylabel('Height (km)')
plt.show()
# Printing results
print("\nLowest density: ")
print(min_density)

print("At altitude: ")
print(h_max_range[min_alt])

# Problem 2
xData = np.array([0.0, 1.0, 2.0, 3.0])
yData = np.array([1.0, 1.0, 0.5, 0.0])

k = curvature(xData,yData)
print("\nNatural case:")
print(k)

x = 2.6
print("y =",evalSpline(xData,yData,k,x))

# created slope function based on curvature function
k = slope(xData,yData)
print("\nClamped case: ")
print(k)
x = 2.6
print("y =",evalSpline(xData,yData,k,x))

# Check using SciPy CubicSpline Function
spline = CubicSpline(xData,yData, bc_type='natural')
prediction = spline(2.6)
print("\nNatural")
print(prediction)

spline = CubicSpline(xData,yData, bc_type='clamped')
prediction = spline(2.6)

print("\nClamped")
print(prediction)

# Problem 3
xData = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
yData = np.array([0.49, 1.6, 3.36, 6.44, 10.16])

xlog = np.log(xData)
ylog = np.log(yData)

m = 1

coeff = polyFit(xlog,ylog,m)

print("Coefficients are: \n",coeff)
print("Std. deviation = ", stdDev(coeff,xlog,ylog))


x1 = min(xData)
x2 = max(xData)
dx = (x2-x1)/20.0
x = np.arange(x1,x2 + dx/10.0,dx)
y = np.zeros((len(x)))*1.0
r = np.zeros(len(x))
R = np.zeros(len(x))
S = 0
for i in range(m):
    y = y + coeff[i]*x**i
    r[i] = yData[i] - ylog[i]
    R[i] = r[i]/yData[i]
    S = S + y[i]**2 +R[i]**2

print(S)
coeff = polyFit(xlog,ylog + S,m)

print("Weighted Coefficients are: \n",coeff)
print("Std. deviation = ", stdDev(coeff,xlog,ylog + S))

plt.loglog(xlog, ylog, marker='o', color='black')
plt.loglog(xlog, 0.53*xlog**(1.88))
plt.loglog(xlog, 0.53*xlog**(1.88) + S, color='red')
plt.legend(['Actual','No weighting', 'Weighted'])
plt.xlabel('x, log scale')
plt.ylabel('y, log scale')
plt.grid (True)
plt.show()


