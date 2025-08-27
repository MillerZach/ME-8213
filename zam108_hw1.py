import numpy as np
import math

# Problem 1a
# count number of students whose overall grade is greater than 60

# load .dat file as first name, last name, grade
grade = np.loadtxt('Homework1Problem1.dat', usecols= (2,), dtype=float, unpack=True)
last, first = np.loadtxt('Homework1Problem1.dat', usecols= (0,1,), dtype=str, unpack=True)

count = 0
for i in range(0,len(grade)):
    if grade[i] > 60:
        count = count + 1

print(count)

# Problem 1b
# count number of students who made a B on the assignment ( 80 <= grade < 90)

count = 0
for i in range(0,len(grade)):
    if np.logical_and(grade[i] >= 80, grade[i] < 90):
        count = count + 1
print(count)

# Problem 1c
# Print names of students who made an A
A_list = ['']*len(grade)
c = 0
for i in range(0,len(grade)):
    if grade[i] >= 90:
        A_list[c] = [first[i] + ' ' +last[i]]
        c = c+1

print(A_list)

# Problem 2:
# Write a file named primes500.dat, with all prime numbers less than 500 written in it. (one number per line)
# Prime number = nondivisible by whole number other than itself
def is_prime(n):
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))

a = np.arange(1, 500)
foo = np.vectorize(is_prime)
pbools = foo(a)
primes = np.extract(pbools, a)
#primes = int(primes)
print(primes)
np.savetxt('Homework1Problem2.dat', primes, delimiter=',')

# Problem 3a
# Find the dot product of a = (1 1 0) and b = (2 -1 3)
a = (1, 1, 0)
b = (2, -1, 3)

print(np.dot(a, b))

# Problem 3b:
# find the outer product of a = (1 1 0) and b = (2 -1 3)
print(np.outer(a, b))

# Problem 4:
# Amicable pair: a pair of positive integers where sum of positive divisors of the first number is equal to the second number
# and the sum of the positive divisors of the second number is equal to the first. Find all such numbers < 500.

for n in range(1, 500):
    #print(f"\nFirst number: {n}")
    def get_divisors(n):
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        if n in divs:
            divs.remove(n)
        return sorted(list(divs))
    #print(f"First number's divisors: {get_divisors(n)}")
    m = sum(get_divisors(n))
    #print(f"Potential second number: {m}")
    l = sum(get_divisors(m))
    #print(f"Sum of second number's divisors: {l}")
    if l == n:
        print(f"Pair found: {n} and {l}")