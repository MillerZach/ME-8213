import numpy as np
import math

# Problem 1a
# count number of students whose overall grade is greater than 60

# load .dat file as variables first name, last name, grade
grade = np.loadtxt('Homework1Problem1.dat', usecols= (2,), dtype=float, unpack=True)
last, first = np.loadtxt('Homework1Problem1.dat', usecols= (0,1,), dtype=str, unpack=True)

# Variable counts number of passing grades
count = 0

# Loop iterates through each grade in the array
for i in range(0,len(grade)):
    # If statement causes count to increase by one if a passing grade is found
    if grade[i] > 60:
        count = count + 1

print(count)

# Problem 1b
# count number of students who made a B on the assignment ( 80 <= grade < 90)

# Variable counts the number of B's
count = 0

# Loop iterates through each grade in the grade array
for i in range(0,len(grade)):

    # np.logical_and requires two logical criteria to be true.
    # If statement adds to count if a grade is found that is greater than or equal to 80 & less than 90
    if np.logical_and(grade[i] >= 80, grade[i] < 90):
        count = count + 1

print(count)

# Problem 1c
# Print names of students who made an A

# Initializing empty string array that is the length of the grade array
A_list = ['']*len(grade)

# index for A_list
c = 0

# Loop iterates through each grade, finds indices for any > 90, stores first and last name of that index
for i in range(0,len(grade)):
    if grade[i] >= 90:

        # combines first and last names to the c'th row of the A_list array
        A_list[c] = [first[i] + ' ' +last[i]]
        c = c+1

print(A_list)

# Problem 2:
# Write a file named primes500.dat, with all prime numbers less than 500 written in it. (one number per line)
# Prime number = nondivisible by whole number other than itself
def is_prime(n):
    # Checks if number is even and greater than 2
    if n % 2 == 0 and n > 2:
        return False
    # all returns true if n % i yields a remainder, false if not.
    # checks for numbers between 3 and the rounded square root + 1 of the number n
    return all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))

a = np.arange(1, 500)

# Using is_prime on each element in a
primes = np.extract(np.vectorize(is_prime)(a), a)

for i in range(0, len(primes)):
    print(primes[i])

np.savetxt('Homework1Problem2.dat', primes, delimiter='\n')

# Problem 3a
# Find the dot product of a = (1 1 0) and b = (2 -1 3)
a = (1, 1, 0)
b = (2, -1, 3)

# using np.dot to compute dot product
print(np.dot(a, b))

# Problem 3b:
# find the outer product of a = (1 1 0) and b = (2 -1 3)

# using np.outer to compute the outer product
print(np.outer(a, b))

# Problem 4:
# Amicable pair: a pair of positive integers where sum of positive divisors of the first number is equal to the second number
# and the sum of the positive divisors of the second number is equal to the first. Find all such numbers < 500.

# This function checks for numbers from 1 to half of n + 1 that divide evenly into n, excluding n.
def get_divisors(n):
    divs = set()
    for i in range(1, int(np.sqrt(n) + 1)):
        # Check for remainder
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    # remove n
    if n in divs:
        divs.remove(n)
    return sorted(list(divs))

# n is potential first number
for n in range(1, 500):

    # Potential second number
    m = sum(get_divisors(n))

    # Sum divisors of potential second number
    l = sum(get_divisors(m))

    # If the sum of the second number's divisors equals the first number, print n (first number) and m (second)
    if l == n:
        print(f"Pair found: {n} and {m}")
