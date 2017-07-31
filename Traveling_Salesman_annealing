#Traveling Salesman Problem
#Michael Toce
#Modeling Analysis
#Assignment1 Problem 3

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools as it

# defining variables

N = 12	# number of cities + 2
n = N-2	# number of cities in Cn
a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
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
xlist = [C[0][0]]
ylist = [C[0][1]]
# move function

def move(Cn):
	np.random.seed()
	Cn1 = np.random.randint(0, len(Cn))
	Cn2 = np.random.randint(0, len(Cn))
	if Cn1 == Cn2:
		Cn2 = Cn1 + 1
		if Cn2 == len(Cn):
			Cn2 = Cn1 - 1
		else:
			Cn2 = Cn1 + 1
	elif Cn1 != Cn2:
		Cn[Cn1][0], Cn[Cn2][0] = Cn[Cn2][0], Cn[Cn1][0]
		Cn[Cn1][1], Cn[Cn2][1] = Cn[Cn2][1], Cn[Cn1][1]
	
	return Cn



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
			L_sum = 0
			L_first = 0
			L_last = 0
			break
	return E

def graph(xlist, ylist, Best):
	for deeznuts in range(0, len(Best)):
		if deeznuts == len(Best)-1:
			xlist.append(C[0][0])
			ylist.append(C[0][1])
			deeznuts += 1
		else:
			xlist.append(Best[deeznuts][0])
			ylist.append(Best[deeznuts][1])
			deeznuts += 1
	lines = plt.plot(xlist, ylist, 'bo', C[0][0], C[0][1], 'g^')
	plt.setp(lines, color='r', ls='-')
	plt.xlim(0, N)
	plt.ylim(0, N)
	return Best

# move function for robust calculation

maxTemp = 10
Temp = maxTemp
minTemp = .1
e = np.math.exp(1)
#print(e)
Enew = 0
Enew = []
Elist = []
clist = []
# lets do this
E = cost(C, Cn)
print(E)
o = 0
while Temp > minTemp and Temp <= maxTemp:
	for c in range(0,10):
		#print(E)
		Cnew = move(Cn)
		Enew = cost(C, Cnew)
		#print(Enew)
		p = np.random.random()
		P = e**(-(Enew-E)/(Temp))
		#print(P)
		if p <= P:
			Elist.append(Enew)
			clist.append(o)
			E = Enew
		else:
			E = cost(C, Cnew)
		Best = Cn
		c += 1
		o += 1
	Temp = 0.5 * Temp

x_list = [C[0][0]]
y_list = [C[0][1]]
print(Best)
for i in range(0, len(Best)):
	x_list.append(Best[i][0])
	y_list.append(Best[i][1])
	i += 1
x_list.append(C[0][0])
y_list.append(C[0][1])
lines = plt.plot(x_list, y_list)
plt.setp(lines, color ='b', ls='-')
plt.xlim(0,N)
plt.ylim(0,N)
plt.plot(x_list, y_list)
plt.show()

Emin = Elist[len(Elist)-1]
print(Emin)
lines = plt.plot(clist, Elist, 'bo')
plt.setp(lines, color='b', ls='-')
plt.xlim(0, len(clist)-1)
plt.ylim(2*N, 7*N)
plt.xlabel('Iteration')
plt.ylabel('Total Path Distance')
plt.title(['The minimum total path length is: ', Emin])
plt.plot(clist, Elist)
plt.show()



