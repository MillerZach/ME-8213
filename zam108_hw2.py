from cmath import isnan

import numpy as np
# pivot example from canvas
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

#a=np.array([[8.0,-6.0,2.0],[-4.0,11.0,-7.0],[4.0,-7.0,6.0]])
#b=np.array([28.0,-40.0,33.0])

a = np.array(([3.0, -1.0, 0.0, 0.0],[0.0,0.0,-1.0,1.0],[0.0,-1.0,3.0,-1.0],[-1.0, 3.0, -1.0, 0.0]))
b = np.array([1.0, 0.0, 0.0, 0.0])
acopy = a.copy()
bcopy = b.copy()

if np.any(np.isnan(gaussElimin(a,b))):
    print(a)
    print("row swap:")

    a = swapRows(a,1,2)
    b = swapRows(b,1,2)

    print(a)
    print(b)

print(gaussElimin(a,b))
#print(a)
#print(acopy)


'''c=np.array([[2.,1.],[2.,1.001]])
d=np.array([3.,0.])
print(gaussElimin(c,d))'''

