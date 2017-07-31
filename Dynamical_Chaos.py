from math import sin, cos
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import newton
from scipy.misc import imresize
import matplotlib.pyplot as pl
import pylab as plb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.cm as cm
import cv2
import Image
from numpy import *
from scipy.misc import lena,imsave
import math
from PIL import Image

# Logistic Map function
def Logistic_Map():

	# Initial X value
	X = 0.1

	# Initialize arrays
	XA = []
	FA = []
	GI = []
	GI2 = []
	IA = []

	# Rate of change
	R = 3

	# Iteration count total
	I = 500

	# Plot X array against itself (1)
	'''pl.figure("Geometric Iteration", figsize = (8, 8))
	pl.title("Geometric Iteration")
	pl.plot(X, 0, color = "Red", label = "Geometric Iteration")'''

	# Calculate and store X value of each iteration
	for i in range (I):
		XA.append(X)
		X = R * X * (1 - X)
		FA.append(X)
		IA.append(i)
		GI.append(XA[i])
		GI.append(XA[i])
		GI2.append(XA[i])
		GI2.append(FA[i])
	
	# Geometric Iteration broken - needs to be fixed
	# Plot X array against itself (2)
	'''pl.plot(GI, GI2, color = "Red")
	pl.plot(XA, XA, color = "Green", label = "f(x) = x")
	pl.plot(XA, FA, ",", color = "Red", label = "Logistic Map")
	pl.xlabel("x value")
	pl.ylabel("f(x) value")
	pl.legend()'''

	# Plot X array against corresponding iteration
	pl.figure("Chaos of Feigenbaum Series", figsize = (8, 8))
	pl.title("Chaos of Feigenbaum Series")
	pl.plot(IA, XA, color='Red', label = "$\mu = 3$")
	pl.xlabel("Iterations")
	pl.ylabel("X-value")
	pl.legend()
	pl.show()

# Logistic bifurcation
def Logistic_Bifurcation():
	
	# Iteratation count total
	I = 4000

	# Truncator
	T = 1000

	# Rate of change
	R = np.linspace(1, 4, I)
	
	# X value
	X = np.ndarray((I,I))
	X[0].fill(0.1)

	# Calculate and X value of each iteration for a given R
	for i in range (I - 1):
		X[i + 1] = R * X[i] * (1 - X[i])
	
	# Reform arrays
	X = (X[(I-T):I]).reshape((1,T*I)).T
	R = np.tile(R, T)
	
	# Plot bifurcation
	pl.figure("Logistic Bifurcation", figsize = (8, 8))
	pl.title("Feigenbaum's Map")
	pl.plot(R, X, ",", color = "Blue")
	pl.xlabel('$\mu$')
	pl.ylabel("x value")

# Arnold's Cat Map function
def Arnold_Cat_Map():
	
	# load image
	im = array(Image.open("Geass.png"))
	print(im)
	N = im.shape[0]
	print(N)
	# create x and y components of Arnold's cat mapping
	x,y = meshgrid(range(N),range(N))
	xmap = (2*x+y) % N
	ymap = (x+y) % N

	for i in xrange(N):
		imsave("GeassMap/geass_{0}.png".format(i),im)
		im = im[xmap,ymap]
	
def Taylor_Greene_Chirikov(K):

	# Iteration count total
	I = 500

	# Angle
	A = np.ndarray((I, I))
	A[0].fill(np.pi)

	# Angular momentum and array
	M = np.ndarray((I, I))
	M[0] = np.linspace(0, 2 * np.pi, 500)

	# Calculate system change over time 
	for i in range (I - 1):
		M[i + 1] = (M[i] + K * np.sin(A[i]))%(2 * np.pi)
		A[i + 1] = (A[i] + M[i + 1])%(2 * np.pi)

	# Plot standard map
	colors = iter(cm.rainbow(np.linspace(0, 2 * np.pi, len(A))))
	pl.figure(K * 100, figsize = (8, 8))
	pl.title("Taylor-Greene-Chirikov Phase Diagram for K = %.2f" % K)
	pl.plot(A, M, ",")
	pl.xlabel("Angle (Radians)")
	pl.ylabel("Angular momentum")
	pl.xlim([0, 2 * np.pi])
	pl.ylim([0, 2 * np.pi])
	#pl.show()
	pl.savefig("TGC/%03d.png" %(K*100))

# Ask which function to run
print "Logistic map           --> Enter 1"
print "Logistic bifurcation   --> Enter 2"
print "Arnold's Cat Map       --> Enter 3"
print "Taylor-Greene-Chirikov --> Enter 4"
Choice = input()

# Run corresponding function
if Choice == 1: Logistic_Map()
if Choice == 2: Logistic_Bifurcation()
if Choice == 3: Arnold_Cat_Map()
if Choice == 4: 
	# Kick parameter
	K = 0
	while (K <= 1):
		Taylor_Greene_Chirikov(K)
		K = K + 0.05
# Diplay Plots
if Choice != 4: pl.show()
