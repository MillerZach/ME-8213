from cmath import isnan

import numpy as np
import math
import time
import matplotlib.pyplot as plt

# pivot example from canvas

# problem 1
import numpy as np
def swapRows(v,i,j):

    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

    return v

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

# gauss elimination example from canvas
def gaussElimin(a,b):

    n = len(b)

    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:

                lam = a [i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]

    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]

    return b

def LUdecomp(a, tol = 1.0e-12):

    n = len(a)
    p = np.arange(n)
    s = np.max(np.abs(a), axis=1)

    for k in range(n-1):
        ratios = np.abs(a[k::n, k]) / s[k:n]
        p_idx = np.argmax(ratios) + k

        if p_idx != k:
            a[[k, p_idx]] = a[[p_idx, k]]
            s[[k, p_idx]] = s[[p_idx, k]]
            p[[k, p_idx]] = p[[p_idx, k]]

        for i in range(k+1, n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i, k+1:n] -= lam*a[k,k+1:n]
                a[i,k] = lam
    return p

def LUsolve(a,b,p):

    n = len(a)
    b[:] = b[p]

    for k in range(1,n):
        b[k] -= np.dot(a[k, :k], b[:k])

    b[n-1] = b[n-1]/a[n-1,n-1]

    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k, k+1:], b[k+1:])) / a[k,k]

    return b

def iterEqs(a,b,x,omega):

    n = len(x)

    for i in range(0,n):

        x[i] = omega*(b[i]-np.sum(np.dot(a[i],x))+a[i,i]*x[i])/a[i,i]+(1-omega)*x[i]

    return x

def gaussSeidel(a,b,iterEqs,x,tol = 1.0e-3):

    omega = 1.0
    k = 10
    p = 1

    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(a,b,x,omega)
        dx = math.sqrt(np.dot(x-xOld,x-xOld))
        if dx < tol: return x,i,omega
        # relaxation factor
        if i == k: dx1 = dx
        if i == k+p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))

    print("Gauss-Seidel failed to converge")

def conjGrad(A,x,b,tol=1.0e-3):
    n = len(b)
    r = b - np.dot(A,x)
    s = r.copy()
    for i in range(n):
        u = np.dot(A,s)
        alpha = np.dot(s,r)/np.dot(s,u)
        x = x + alpha*s
        r = b - np.dot(A,x)
        if(math.sqrt(np.dot(r,r))) < tol:
            break
        else:
            beta = -np.dot(r,u)/np.dot(s,u)
            s = r + beta*s
    return x,i

def matrix_builder(n):
    A=np.zeros((n,n))
    I = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i,j] = 2.0
                I[i,j] = 1.0
            elif np.abs(i-j) == 1.0:
                A[i,j] = -1.0
                I[i,j] = 0.0
            else:
                A[i,j] = 0.0
                I[i,j] = 0.0
    b = np.zeros(n)
    b[n-1] = 1.0
    return A, b, I


#problem 1
print("PROBLEM 1: ")
a=np.array([[3., -1., 0., 0.], [0.,0.,-1.,1.], [0.,-1.,3.,-1.], [-1., 3., -1., 0.]])
b = np.array([1., 0., 0., 0.])
acopy = a.copy()
bcopy = b.copy()

x=gaussPivot(acopy,bcopy)

print(x)


print("\nCheck: Ax")
print(np.dot(a,x))
#print(a)
#print(acopy)

I = np.array([[1.,0.,0.,0.,], [0., 1., 0., 0.,], [0., 0., 1., 0.], [0., 0., 0., 1.]])
Icopy = np.copy(I)

n = a.shape[0]
x = np.zeros((n,n))
for j in range(n):
    e = I[:,j]
    x[:, j] = gaussPivot(a.copy(),e.copy())

print("\n")
print(x)


print("\nCheck: Ax")
print(np.dot(a,x))

# problem 2:

'''a = np.array([[-1., 1., -1/np.sqrt(2), 0., 0., 0.], [0., 0., 1/np.sqrt(2), 1., 0., 0.], [0., -1., 0., 0., -1/np.sqrt(2), 0.], [0., 0., 0., 0., 1/np.sqrt(2), 0.], [0., 0., 0., 0., 1/np.sqrt(2), 1.], [0., 0., 0., -1., -1/np.sqrt(2), 0.]])
b = np.array([0., 18., 0., 12., 0., 0.])

aoriginal = a.copy()
boriginal = b.copy()

#print(np.linalg.matrix_rank(a))
p=LUdecomp(a)
print(p)
x=LUsolve(a,b,p)
print(x)'''
# problem 3:

print("PROBLEM 3:")
A = np.array([[-3., -2., 1., 0., 0., 1.],[-2., 4., -2., 1., 0., 0.,],[1., -2., 4., -2., 1., 0.],[0., 1., -2., 4., -2., 1.],[0., 0., 1., -2., 4., -2.],[1., 0., 0., 1., -2., 3.]])
b = np.array([10., -8., 10., 10., -8., 10.])

# part a: gauss-seidel
print("\n--------------Gauss-Seidel--------------")
x = np.zeros(6)
x,numIter,omega = gaussSeidel(A,b,iterEqs,x)
print('\niter # = ',numIter)
print('Relaxation factor = ',omega)
print('Solution = ',x)

# part b: conjugate gradient
print("\n--------------Conjugate Gradient--------------")
x = np.array([0., 0., 0., 0., 0., 0.])
solution,iter=conjGrad(A,x,b)

print("Solution =",solution)
print("Iterations:",iter)

# problem 4:
print("\nPROBLEM 4:")

# part a:
n = 10
A, b, I = matrix_builder(n)
x = gaussElimin(A,b)
print("\nSolution for n = 10: ")
print(x)

x = np.zeros((n,n))

for j in range(n):
    e = I[:,j]
    x[:, j] = gaussElimin(A.copy(),e.copy())

print("\nInverse of A:")
print(x)

i = 0
stats = np.zeros((20,2))
iterations = (np.logspace(math.log10(2), math.log10(5000), num=5))
for val in iterations:
    n = int(val)
    round(n)
    print(n)
    A, b, I = matrix_builder(n)
    x = np.zeros((n, n))
    start = time.time()
    for j in range(n):
        e = I[:, j]
        x[:, j] = gaussElimin(A.copy(), e.copy())
    elapsed = time.time() - start
    stats[i,0] = n
    stats[i,1] = elapsed
    n = n + 500
    i = i+1

print(stats)

plt.figure(figsize=(8, 6))

linear = np.poly1d(np.polyfit(stats[:,0], stats[:,1], 1))
quadratic = np.poly1d(np.polyfit(stats[:,0], stats[:,1], 2))
cubic = np.poly1d( np.polyfit(stats[:,0], stats[:,1], 3))

y_fit_linear = linear(stats[:,0])
y_fit_quadratic = quadratic(stats[:,0])
y_fit_cubic = cubic(stats[:,0])

plt.scatter(stats[:,0], stats[:,1], marker='o', color='red', label='Actual')
plt.scatter(stats[:,0], y_fit_linear, marker='.', color='green', label='Linear')
plt.scatter(stats[:,0], y_fit_quadratic, marker='.', color='blue', label='Quadratic')
plt.scatter(stats[:,0], y_fit_cubic, marker='.', color='orange', label='Cubic')

plt.grid(True)
plt.legend()

plt.show()



























