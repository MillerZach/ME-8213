import numpy as np

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

a=np.array([[8.0,-6.0,2.0],[-4.0,11.0,-7.0],[4.0,-7.0,6.0]])
b=np.array([28.0,-40.0,33.0])

acopy = a.copy()
bcopy = b.copy()

print(acopy)
print("\n")
print(bcopy)
print("\nx:")
x = gaussElimin(acopy,bcopy)
print(x)
print("\nA:")
print(a)

Ax = np.dot(a,x)
print("Ax:")
print(Ax)


c=np.array([[2.,1.],[2.,1.001]])
d=np.array([3.,0.])
ccopy = c.copy()
dcopy = d.copy()
print(ccopy)
print(dcopy)
print(gaussElimin(c,d))
