import numpy as np
import matplotlib.pyplot as plt

#variables
T = 20			#width/length of city
sigma = 0.5		#population density
H = []
S = []
val = []
f = 0
g = 0
i = 0
j = 0
k = 0
D = 50

def constructpop(f,T,sigma,val,j,k):
	while f < int((T**2)*sigma):
		val = []
		xH = np.random.randint(0,T)
		yH = np.random.randint(0,T)
		vH = 1
		sH = 0
		dice = np.random.random()
		if dice < 0.5:
			gH = 1
		elif dice >= 0.5:
			gH = 2
		val.append(xH)
		val.append(yH)
		val.append(vH)
		val.append(sH)
		val.append(gH)
		for j in range(0,len(H)+1):		#if the point is already used in H, redo this iteration
				if j < len(H):
					if xH == H[j][0] and yH == H[j][1]:
						break
					else:						#if it doesn't conflict, move to next point or if last point then append the value
						j += 1
				else:
					H.append(val)
					f += 1
		f += 1
	for j in range(int(len(H)/2)):
		val = []
		H[j][2] = 2
		for x in range(len(H[j])):
			val.append(H[j][x])
			x += 1
		S.append(val)
		del H[j]
		j += 1
constructpop(0,T,sigma,val,0,0)

def initialplot(i,j,T,sigma,H,S):
	while i < len(H):
		plt.plot(H[i][0],H[i][1],'o',color='yellow',ms=5)	#plots x,y coord of each healthy person
		i += 1
	while j < len(S):
		plt.plot(S[j][0],S[j][1],'o',color='blue',ms=5)	#plots x,y coord of each sick person
		j += 1
	plt.title("Initial Positions: Foxes and Rabbits")
	plt.xlabel("Initial X-position")
	plt.ylabel("Initial Y-position")
	plt.xlim(-1,T+1)
	plt.ylim(-1,T+1)
	plt.show()
initialplot(i,j,T,sigma,H,S)

N = H				#to create a new list of all people in city, will rename H as N
for r in range(len(S)):
	N.append(S[r])		#new list of all people in the city
	r += 1
#print(N)

def move(A,T):
	N = []
	l = 0
	for l in range(0,len(A)):
		Nx = 0
		Ny = 0
		Nv = 0
		onethird = 1.0/3.0
		twothirds = 2.0/3.0
		val = []
		sicktime = []
		xdice = np.random.random()
		ydice = np.random.random()
		if xdice < onethird:						#x moves left
			Nx = A[l][0]-1
		elif xdice > onethird and xdice < twothirds:	#x moves right
			Nx = A[l][0]+1
		elif xdice >= twothirds:					#x stays
			Nx = A[l][0]
		if ydice < onethird:						#y moves up
			Ny = A[l][1]-1
		elif ydice > onethird and ydice < twothirds:	#y moves down
			Ny = A[l][1]+1
		elif ydice >= twothirds:					#y stays
			Ny = A[l][1]
		if Nx == -1:							#wrapping left x
			Nx = T-1
		elif Nx == T:							#wrapping right x
			Nx = 0
		if Ny == -1:							#wrapping up y
			Ny = T-1
		elif Ny == T:							#wrapping down y
			Ny = 0
		Nv = A[l][2]
		#if A[l][2] == 2:						#mark sick people
			#if A[l][3] <= 20:
			#	q = (A[l][3]+1)
			#else:
			#	q = (A[l][3])
			#	Nv = -1
		#else:
			#q = 0
		q = A[l][3]
		g = A[l][4]
		val.append(Nx)
		val.append(Ny)
		val.append(Nv)
		val.append(q)
		val.append(g)
		N.append(val)
		l += 1
	return N
#N = move(A,10)

#print(A)
#print(N)

def checkoverlap(N):
	I = []
	C = []
	m = 0
	n = 0
	for m in range(0,len(N)):
		for n in range(0,len(N)):
			if N[m][0] == N[n][0] and N[m][1] == N[n][1] and m != n:
				v = 0
				val = []
				coord = []
				val.append(N[m][0])
				val.append(N[m][1])
				val.append(N[m][2])
				val.append(N[n][2])
				val.append(N[m][3])
				val.append(N[n][3])
				val.append(N[m][4])
				val.append(N[n][4])
				coord.append(N[m][0])
				coord.append(N[m][1])
				C.append(coord)
				I.append(val)
				n += 1
			else:
				n += 1
		m += 1
	return I, C
I, C = checkoverlap(N)
#print(C)
print(I)

def interaction(C,I,N):
	kc = 0
	t = 0
	for a in range(len(I)):
		if I[a][2] == 1 and I[a][3] == 1 and I[a][6] != I[a][7]:	#if two opposite gender rabbits on same space
			dice1 = np.random.random()
			if dice1 < .75:
				val = []
				val.append(I[a][0])
				val.append(I[a][1])
				val.append(I[a][2])
				val.append(I[a][4])
				dice2 = np.random.random()
				if dice2 < 0.5:
					val.append(1)
				elif dice2 >= 0.5:
					val.append(2)
				print("newrabbit")
				N.append(val)
				a += 1
			else:
				a += 1
		elif I[a][2] != I[a][3]:	#if fox and rabbit on same space
			if I[a][2] == 1:
				k = len(N)
				while t < k:
					#print(N[t][0])
					#print(t)
					#print(len(N)-kc)
					if N[t][0] == I[a][0] and N[t][1] == I[a][1] and N[t][2] == I[a][2]:
						print("kill rabbit")
						#del N[t]	#kill that mofuckin rabbit
						N[t][2] = 0
						k -= 1
						a += 1
						t += 1
					else:
						t += 1
			elif I[a][3] == 1:
				k = len(N)
				while t < len(N):
					#print(N[t][0])
					#print(t)
					#print(len(N)-kc)
					if N[t][0] == I[a][0] and N[t][1] == I[a][1] and N[t][2] == I[a][3]:
						print("kill rabbit")
						#del N[t]	#kill that mofuckin rabbit
						N[t][2] = 0
						k -= 1
						a += 1
						t += 1
					else:
						t += 1
		else:
			a += 1
	return N
#I = interaction(C,I,N)

for d in range(0,D):
	#print("inloop")
	print(len(N))
	Rlist = []
	Flist = []
	Ilist = []
	N = move(N,T)
	I, C = checkoverlap(N)
	N = interaction(C,I,N)
	for r in range(len(N)):
		if N[r][2] == 1:
			Rlist.append(1)
			r += 1
		elif N[r][2] == 2:
			Flist.append(1)
			r += 1
		#elif N[r][2] == -1:
			#Ilist.append(1)
			#r += 1
	plt.plot(d,sum(Rlist), 'o',color='yellow',ms=3)
	plt.plot(d,sum(Flist), 'o',color='blue',ms=3)
	#plt.plot(d,sum(Ilist), 'o',color='green',ms=3)
	d += 1
plt.title("Population Over Time: sigma=0.5")
plt.xlabel("Number of Days")
plt.ylabel("Population Size")
plt.xlim(-1,D+1)
plt.ylim(-1,int((T**2)*sigma)/4)
plt.show()

