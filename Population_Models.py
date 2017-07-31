#Michael Toce
#Modeling Analysis
#Assignment 4: Population Models

import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint

year = [1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
pop = [3929214., 5308483., 7239881., 9638453., 12860702., 17063353., 23191876., 31443321., 38558371., 50189209., 62979766., 76212168., 92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203302031., 226542199., 248709873.]

def exponential():
	yr = np.arange(0, 201, 10)
	year = [1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
	pop = [3929214., 5308483., 7239881., 9638453., 12860702., 17063353., 23191876., 31443321., 38558371., 50189209., 62979766., 76212168., 92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203302031., 226542199., 248709873.]

	lnpop = np.log(np.array(pop)/pop[0])
	p = np.polyfit(yr,lnpop,1)
	q = np.polyfit(yr,lnpop,2)
    #plt.figure(1)
    #plt.plot(year, lnpop, 'o', color = 'black')
    #plt.plot(year,p[0]*np.array(yr)+p[1],'b-', label = 'y = %s t + %s ' %(str(round(p[0], 2)), str(round(p[1],2))))
    #plt.plot(year,q[0]*np.array(yr)**2+q[1]*np.array(yr)+q[2], 'g-', label = 'y = %s t^2 + %s t + %s ' %(str(round(q[0], 3)), str(round(q[1],2)), str(round(q[2],2))))
    #plt.xlabel('Year')
    #plt.ylabel('ln(N/No)')
    #plt.legend(loc = 'upper left')
    #print p[0]
    
	plt.figure(1)
	plt.plot(year, pop, 'o', color = 'black')
	popexp1 = pop[0] * np.exp(p[0] * yr)
	popdif1 = abs(pop - popexp1)/pop * 100.
	#popexp2 = pop[0] * np.exp(p[0] * yr + p[1])
	#popdif2 = abs(pop - popexp2)/pop * 100.
	#popexp3 = pop[0] * np.exp(q[0] * (np.array(yr))**2 + q[1]*(np.array(yr))+ q[2])
	#popdif3 = abs(pop - popexp3)/pop * 100.
	#plt.plot(year, popexp3, ls = '--')
	#plt.plot(year, popexp2, ls = '-.')
	plt.plot(year, popexp1, ls = '-')
	plt.xlabel('Year')
	plt.ylabel('Population')
	plt.xlim(1785, 1995)
    
	plt.figure(3)
	plt.plot(year, popdif1, ls = '-')
    #plt.plot(year, popdif2, ls = '-.')
    #plt.plot(year, popdif3, ls = '-')
    #plt.xlim(1785, 1995)
	plt.xlabel('Year')
	plt.ylabel('Percent Error')
	#plt.show()
	return popexp1, popdif1
    
#popexp1, popdif1 = exponential()

def logistic(k, r):
	yr = np.arange(0, 201, 10)
	year = [1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
	pop = [3929214., 5308483., 7239881., 9638453., 12860702., 17063353., 23191876., 31443321., 38558371., 50189209., 62979766., 76212168., 92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203302031., 226542199., 248709873.]
	
	plt.figure(4)
	A = k / pop[0] - 1.
	poplog = k / (1 + A*np.exp(-r * yr))
	plt.plot(year, pop, 'o', color = 'black')
	plt.plot(year, poplog, 'b-')
	popdiflog = abs(pop-poplog)/pop * 100
	plt.xlim(1785, 1995)
	plt.xlabel('Year')
	plt.ylabel('Population')
	
	plt.figure(5)
	plt.plot(year, popdiflog, ls = '-')
	plt.xlabel('Year')
	plt.ylabel('Percent Error')
	plt.show()
	return poplog, popdiflog
'''
poplog, popdiflog = logistic(350000000, 0.027)
plt.figure(6)
plt.plot(year, pop, 'o', color = 'black')
plt.plot(year, poplog, ls = '-', label = 'Logistic')
plt.plot(year, popexp1, ls = ':', label = 'Exponential')
plt.xlim(1785, 1995)
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(loc = 'best')

plt.figure(7)
plt.plot(year, popdiflog, ls = '-', label = 'Logistic')
plt.plot(year, popdif1, ls = ':', label = 'Exponential')
plt.xlim(1785, 1995)
plt.xlabel('Year')
plt.ylabel('Percent Error')
plt.legend(loc = 'best')
plt.show()
'''
def predpos(ic, t, A, B, C, D):
    x = ic[0]
    y = ic[1]
    vx = A * x - B * x * y
    vy = -1. * C * y + B*D * x * y
    return vx, vy
    
def predposw(ic, t, A, B, C, D, E, F):
    x = ic[0]
    y = ic[1]
    z = ic[2]
    vx = A * x - B * x * y
    vy = -C * y * z + B*D * x * y
    vz = -1. * E * z + F*y*z
    return vx, vy, vz

def predpreyw(x, y, z, A, B, C, D, E, F):
    t = np.linspace(0,6,1000)
    print len(t)
    ic = [x, y, z]
    pops = odeint(predposw, ic, t, args=(A, B, C, D, E, F))
    print ic
    print len(pops)
    plt.figure(1)
    plt.plot(t, pops[:,0], 'b-', label = 'Rabbit')
    plt.plot(t, pops[:,1], 'r-', label = 'Fox')
    plt.plot(t, pops[:,2], 'g-', label = 'Wolf')
    plt.legend(loc = 'upper right')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
#predpreyw(10., 5., 1., 1., 1., 1., 1., 1., 1.)

def predprey(x, y, A, B, C, D):
    t = np.arange(0., 10., 0.01)
    jac = np.array([[A - B*y, -B*x], [B*D*y, -C+B*D*x]])
    lambda1, lambda2 = np.linalg.eigvals(jac)
    T = 2*np.pi/abs(lambda1)
    print T, lambda1
    t = np.linspace(0,15,1000)
    print len(t)
    ic = [x, y]
    pops = odeint(predpos, ic, t, args=(A, B, C, D))
    print ic
    print len(pops)
    plt.figure(1)
    plt.plot(t, pops[:,0], 'b-', label = 'Rabbit')
    plt.plot(t, pops[:,1], 'r-', label = 'Fox')
    plt.legend(loc = 'upper right')
    plt.xlabel('Time')
    plt.ylabel('Population')
    cons = [[1., 1.1, 1.2],[0.125, 0.15, 0.2],[1.4, 1.5, 1.6],[0.65, 0.75, 0.85]]
    popula = [[],[]]
    populb = [[],[]]
    populc = [[],[]]
    populd = [[],[]]
    #popul = [[],[]]
    plt.figure(2)
    for i in range(len(cons[0])):
        q = odeint(predpos, ic, t, args=(cons[0][0], cons[1][i], cons[2][0], cons[3][0]))
        populb[0].append(q[:,0])
        populb[1].append(q[:,1])
        plt.plot(populb[0][i], populb[1][i])
        plt.xlabel('Rabbit Population')
        plt.ylabel('Fox Population')
    plt.figure(3)
    for i in range(len(cons[0])):
        q = odeint(predpos, ic, t, args=(cons[0][i], cons[1][0], cons[2][0], cons[3][0]))
        popula[0].append(q[:,0])
        popula[1].append(q[:,1])
        plt.plot(popula[0][i], popula[1][i])
        plt.xlabel('Rabbit Population')
        plt.ylabel('Fox Population')
    plt.figure(4)
    for i in range(len(cons[0])):
        q = odeint(predpos, ic, t, args=(cons[0][0], cons[1][0], cons[2][i], cons[3][0]))
        populc[0].append(q[:,0])
        populc[1].append(q[:,1])
        plt.plot(populc[0][i], populc[1][i])
        plt.xlabel('Rabbit Population')
        plt.ylabel('Fox Population')
    plt.figure(5)
    for i in range(len(cons[0])):
        q = odeint(predpos, ic, t, args=(cons[0][0], cons[1][0], cons[2][0], cons[3][i]))
        populd[0].append(q[:,0])
        populd[1].append(q[:,1])
        plt.plot(populd[0][i], populd[1][i])
        plt.xlabel('Rabbit Population')
        plt.ylabel('Fox Population')
    ipop = [[10, 11, 12, 13, 15, 18, 20, 25],5.01]
    for i in range(len(ipop[0])):
        plt.figure(7)
        #popul[0].append(q[:,0])
        #popul[1].append(q[:,1])
        ip = [ipop[0][i], ipop[1]]
        q = odeint(predpos, ip, t, args=(A,B,C,D))
        plt.plot(q[:,0], q[:,1])
        plt.xlabel('Rabbit Population')
        plt.ylabel('Fox Population')
    print ic
    plt.figure(2)
    plt.plot(t, popula[0][0])
    plt.plot(t, popula[1][0])
    plt.plot(popula[0][0], popula[1][0])
    plt.show()
    
#predprey(10., 5., 1., .2, 1.5, .75)
#predprey(10., 5., 1., .1, 1.5, .75)

def epipos(ic, t, a, b, c):
    H = ic[0]
    S = ic[1]
    I = ic[2]
    #V = ic[3]
    #ds = -a * S * I - c * S
    dH = -a * H * S - c * H
    #di = a * S * I - b * I
    dS = a * H * S - b * S
    #dr = b * I
    dI = c * H + b * S
    #dv = c * S
    return dH, dS, dI

def epi(H, S, I, a, b, c):
    t = np.linspace(0,30,1000)
    ic = [H, S, I]
    pops = odeint(epipos, ic, t, args=(a, b, c))
    sumpops = pops[:,0] + pops[:,1] + pops[:,2]
    plt.plot(t, pops[:,0], 'b-', label = 'Healthy')
    plt.plot(t, pops[:,1], 'r-', label = 'Sick')
    plt.plot(t, pops[:,2], 'g-', label = 'Immune')
    #plt.plot(t, pops[:,3], color = 'purple', label = 'Vaccinated')
    plt.plot(t, sumpops, ls = ':', label = 'Sum')
    plt.legend(loc = 'best')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

epi(100., 1., 0, .005, .1, .05)
