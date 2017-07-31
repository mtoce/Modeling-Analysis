#Traveling Salesman Problem
#Michael Toce
#Modeling Analysis
#Assignment1 Problem 3

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools as it

# defining variables

N = 5	# number of cities
a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
x = 0
y = 0
r1 = 0
r2 = 0

# Randomized X and Y coordinates for cities

X = []
Y = []
Xn = []
Yn = []
def city_maker(X, Y):
	np.random.seed(69)	# seed 69
	for a in range(0, N+1):
		if a == 0:
			x = np.random.random() * N
			X.append(x)
			x = 0
			y = np.random.random() * N
			Y.append(y)
			y = 0
		elif a > 0 and a < N:
			x = np.random.random() * N
			X.append(x)
			Xn.append(x)
			x = 0
			y = np.random.random() * N
			Y.append(y)
			Yn.append(y)
			y = 0
		else:
			x = X[0]
			X.append(x)
			y = Y[0]
			Y.append(y)
			break
	return X, Y
city_maker(X, Y)
C = [X, Y]	# each array has 1 more element than the number of cities so we can return to the same first city at the end
Cn = [Xn, Yn]
print(C)
print(Cn)
'''	
plt.xlim(0, N)
plt.ylim(0, N)
plt.plot(X, Y)
plt.show()
print(X, Y)
# move function
'''
'''
def move(r1, r2):
	np.random.seed()
	r1 = np.random.randint(1, N)
	r2 = np.random.randint(1, N)
	if r1 == r2:
		r2 = r1+1
		if r2 == N+1:
			r2 = r1-1
		else:
			r2 = r1+1
	elif r1 != r2:
		X[r1], X[r2] = X[r2], X[r1]
		Y[r1], Y[r2] = Y[r2], Y[r1]
	
	return r1, r2

r1, r2 = move(r1, r2)

print(r1)
print(r2)
print(X, Y)
'''
# cost function
La = []	# array of all distances in a certain permutation
E = 0
L = 0

def cost(L, La, E):
	for b in range(0, N):
		L = np.math.sqrt((X[b+1]-X[b])**2+(Y[b+1]-Y[b])**2)
		#print(L)
		La.append(L)
		b += 1
	E = sum(La)
	return L, La, E
L, La, E = cost(L, La, E)
print(L)
print(E)

# move function for robust calculation

Xmin = 0
Ymin = 0
P = np.math.factorial(N)
#def robust()
	
'''
def robust_move(X, Y, L, La, La2, E, Xmin, Ymin):
	c = 1
	for c in range(1, N-c+1):
		print(1)
		for d in range(0, N): 
			print(2)
			if c == 1 and d == 0:	# needed to make sure we include the initial distance in calculations
				for e in range(0, N):
					print(3)
					L = np.math.sqrt((X[e+1]-X[e])**2+(Y[e+1]-Y[e])**2)
					La.append(L)
					e += 1
				d += 1
			else:
				print(4)
				X[d], X[d+1] = X[d+1], X[d]	# swaps the 1st x variable with the next, then the next, then the next...
				Y[d], Y[d+1] = Y[d+1], Y[d]	# ''y variable''
				L, La, La2, E = cost(L, La, La2, E)
				if E < La2[f-1]:
					print(5)
					# print(E)
					# print(La2[c*d-1])
					Xmin = X
					Ymin = Y
					d += 1
				else:
					print(c)
					print(d)
					print(f-1)
					print(La2[f-1])
					print(6)
					d += 1
		c += 1
X, Y, L, La, La2, E, Xmin, Ymin = robust_move(X, Y, L, La, La2, E, Xmin, Ymin)
'''
# robust way of determining shortest distance
'''
P = np.math.factorial(N)	# number of different ways we can travel
print(P)
def robust_shortest_distance():
	for e in range(0, P):
'''
