#Traveling Salesman Problem
#Michael Toce
#Modeling Analysis
#Assignment1 Problem 3

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools as it

# defining variables

N = 15	# number of cities
a = 0
b = 0
c = 0
d = 0
x = 0
y = 0


# Randomized X and Y coordinates for cities

C = []
Cn = []
def city_maker(C, Cn):
	np.random.seed(25)	# seed 69
	for a in range(0, N):
		if a == 0:
			x = np.random.random() * N
			y = np.random.random() * N
			C.append([x, y])
			x = 0
			y = 0
		elif a > 0 and a < N-1:
			x = np.random.random() * N
			y = np.random.random() * N
			C.append([x, y])
			Cn.append([x, y])
			x = 0
			y = 0
		else:
			C.append(C[0])
			break
	return C, Cn
C, Cn = city_maker(C, Cn)
#print(C)
#print(Cn)

# cost function
La = []	# array of all distances in a certain permutation
E = 0
L = 0
def cost(C, Cn):
	Llist = []
	for b in range(0, len(Cn)):
		if b < len(Cn)-1:
			L = np.math.sqrt((Cn[b+1][0]-Cn[b][0])**2+(Cn[b+1][1]-Cn[b][1])**2)	# subtracts x's from x's and y's from y's to find the distance between all points for a certain permutation
			#print("L =", L)
			Llist.append(L)
			b += 1
		elif b == len(Cn)-1:
			L_first = np.math.sqrt((Cn[0][0]-C[0][0])**2+(Cn[0][1]-C[0][1])**2)
			#print("L_first = ", L_first)
			L_last = np.math.sqrt((C[0][0]-Cn[len(Cn)-1][0])**2+(C[0][1]-Cn[len(Cn)-1][1])**2)
			#print("L_last = ", L_last)
			L_sum = sum(Llist)
			#print("L_sum = ", L_sum)
			E = L_sum + L_first + L_last
			Llist = []
			break
	return E
E = cost(C, Cn)
#print(Llist)
#print(E)
#print(E)

# move function for robust calculation

P = np.math.factorial(len(Cn))
Ea = []
I = list(it.permutations(Cn))
def robust(I, Ea):
	for c in range(0, P):
		#print(I)
                if c == 0:
                        E = cost(C, I[c])
			#print("E1 = ", E)
                        c += 1
               	else:
                        Enew = cost(C, I[c])
			#print("Enew1 =", Enew)
                        if Enew <= E:
                                #print("Enew = ", Enew)
                                Best = I[c]
				Ea.append(Enew)
                                E = Enew
                                c += 1
                        else:
				#print("E = ", E)
                                c += 1
                
		#Ea.append(cost(C, I[c]))
		#c += 1
		
	return E, Best
'''
E, Best = robust(I, Ea)
xlist = [C[0][0]]
ylist = [C[0][1]]
for deeznuts in range(0, len(Best)):
	if deeznuts == len(Best)-1:
		xlist.append(C[0][0])
		ylist.append(C[0][1])
		deeznuts += 1
	else:
		xlist.append(Best[deeznuts][0])
		ylist.append(Best[deeznuts][1])
		deeznuts += 1
'''
Emin = Ea[len(Ea)-1]
print(Emin)
'''
#print(xlist, ylist)
lines = plt.plot(xlist, ylist)
plt.setp(lines, color='r', ls='-')
plt.xlim(0, N)
plt.ylim(0, N)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.plot(xlist, ylist, 'bo', xlist[0], ylist[0], 'g^')
plt.title(['The minimum path length is: ', Emin])
plt.show()
'''
