#Michael Toce
#Modeling Analysis
#Assig3: The Kepler Problem

import numpy as np

a = 0.587	# semimajor axis

Pyear = 76	# period of orbit in years
Psec = Pyear*365.25	# period of orbit in seconds
t = float(raw_input("What time (JD) do you want to calculate the position of Halley's comet?"))
T = 2446470.50000

#1: Iterative Solution

def derivative(f, x, h):
      return (f(x+h) - f(x-h)) / (2.0*h)  # might want to return a small non-zero if ==0

def quadratic(E):
	e = 0.967	# eccentricity
	M = (2*np.math.pi*(t-T))/Psec
    	return E-(e*np.math.sin(E))-M     # just a function to show it works

def solve(f, x0, h):
    	lastX = x0
    	nextX = lastX + 10 * h  # "different than lastX so loop starts OK
    	while (abs(lastX - nextX) > h):  # this is how you terminate the loop - note use of abs()
        	newY = f(nextX)                     # just for debug... see what happens
        	print "f(", nextX, ") = ", newY     # print out progress... again just debug
        	lastX = nextX
        	nextX = lastX - newY / derivative(f, lastX, h)  # update estimate using N-R
    	return nextX

xFound = solve(quadratic, 1, 0.01)    # call the solver
print "solution: x = ", xFound        # print the result
