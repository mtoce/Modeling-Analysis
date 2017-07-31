import matplotlib as plt
from math import sin, cos, pi
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import newton
from pylab import *

N = 1000
y = np.zeros([4])

# define constants
L_0 = 0.5		#initial length of string
L_0_2 = 0.75
L_0_3 = 1.0
L_0_4 = 1.25
L_0_5 = 1.5
L_0_6 = 1.01
deltaL = 0.1	#initial stretch of string
v_0 = 0.0		#initial velocity
rad = pi/180	#radians in 1 deg
theta_0 = 0.5	#initial angle of pendulum in radians
theta_0_2 = 0.75
theta_0_3 = 1.0
theta_0_4 = 1.25
theta_0_5 = 1.5
omega_0 = 0.0	#inital angular velocity of pendulum

P = 100
time = linspace(0,P,N)

k = 1.5			#spring constant
m = 0.2			#mass of bob
g = 9.8			#acceleration due to gracity in SI units

# without spring

f0 = np.array([theta_0,omega_0])
f0_2 = np.array([theta_0_2,omega_0])
f0_3 = np.array([theta_0_3,omega_0])
f0_4 = np.array([theta_0_4,omega_0])
f0_5 = np.array([theta_0_5,omega_0])

def simple_pendulum_deriv(f,time):		
	
	dtheta = f[1]
	domega = -m*g*L_0_3*sin(f[0])
	dfdt = [dtheta, domega]
	
	return dfdt
	
def simple_pendulum_deriv2(f,time):
	dtheta = f[1]
	domega = -m*g*L_0_6*sin(f[0])
	dfdt = [dtheta, domega]
	
	return dfdt
	
ans1 = odeint(simple_pendulum_deriv, f0, time)
ans2 = odeint(simple_pendulum_deriv, f0_2,time)
ans3 = odeint(simple_pendulum_deriv, f0_3,time)
ans4 = odeint(simple_pendulum_deriv, f0_4,time)
ans5 = odeint(simple_pendulum_deriv, f0_5,time)
ans6 = odeint(simple_pendulum_deriv2, f0_3,time)
xpos3 = L_0_3*sin(ans3[:,0])
xpos6 = L_0_6*sin(ans6[:,0])
ypos3 = -L_0_3*cos(ans3[:,0])
ypos6 = -L_0_6*cos(ans6[:,0])
TH1 = ans1[:,0]
TH2 = ans2[:,0]
TH3 = ans3[:,0]
TH4 = ans4[:,0]
TH5 = ans5[:,0]
OM1 = ans1[:,1]
OM2 = ans2[:,1]
OM3 = ans3[:,1]
OM4 = ans4[:,1]
OM5 = ans5[:,1]

# graph the pos
plt.plot(xpos3,ypos3, 'b-')
plt.xlabel("X-Position")
plt.ylabel("Y-Position")
plt.xlim(-1.2,1.2)
plt.ylim(-1.2,1.2)
plt.title("Trajectory of Pendulum Without Spring")
plt.show()

#graph the energy
T = 0.5*m*(L_0_3**2)*ans3[:,1]**2
U = m*g*L_0_3*ypos3
Lagrangian = T - U
Hamiltonian = T + U
th = np.arange(0,2.9,0.2)
Tharray = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
Parray = [2.249,2.267,2.296,2.337,2.393,2.465,2.555,2.667,2.807,2.983,3.208,3.502,3.908,4.5345,5.772]
Psmall = 2*pi*np.sqrt(L_0_3/g)

#graph the total energy as a function of time
plt.plot(time,Hamiltonian)
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.ylim(-5,5)
plt.title("Total Energy vs. Time (Without Spring)")
plt.show()

#graph the period as a function of angle
plt.plot(Tharray,Parray, 'o')
plt.axhline(y = Psmall)
plt.xlabel("Theta")
plt.ylabel("Period")
plt.ylim(0,6.0)
plt.title("Period vs. Theta")
plt.show()

#graph the chaos of the system
plt.plot(time,abs((xpos3**2+ypos3**2)**2-(xpos6**2+ypos6**2)**2), 'c-',lw = 1)
plt.xlabel("Time")
plt.ylabel("Difference in Length")
#plt.ylim(0,.08)
plt.title("Dynamical Chaos vs. Time of Simple Pendulum")
plt.show()

#graph phase plot
plt.plot(TH1,OM1, 'c-',lw = 1)
plt.plot(TH2,OM2, 'm-',lw = 1)
plt.plot(TH3,OM3, 'b-',lw = 1)
plt.plot(TH4,OM4, 'g-',lw = 1)
plt.plot(TH5,OM5, 'y-',lw = 1)
plt.xlabel("Theta")
plt.ylabel("Omega")
plt.title("Phase Plot: Theta vs. Omega of Simple Pendulum")
plt.show()

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

# with spring

# set initial state
y1 = np.array([deltaL,v_0,theta_0,omega_0])
y2 = np.array([deltaL,v_0,theta_0_2,omega_0])
y3 = np.array([deltaL,v_0,theta_0_3,omega_0])
y4 = np.array([deltaL,v_0,theta_0_4,omega_0])
y5 = np.array([deltaL,v_0,theta_0_5,omega_0])

def with_spring(y,time):
	
	g0 = y[1]
	g1 = (L_0_3+y[0])*y[3]*y[3] - k/m*y[0] + g*cos(y[2])
	g2 = y[3]
	g3 = -(g*sin(y[2]) + 2.0*y[1]*y[3])/(L_0_3+y[0])
	
	return np.array([g0,g1,g2,g3])

def with_spring2(y,time):
	
	g0 = y[1]
	g1 = (L_0_6+y[0])*y[3]*y[3] - k/m*y[0] + g*cos(y[2])
	g2 = y[3]
	g3 = -(g*sin(y[2]) + 2.0*y[1]*y[3])/(L_0_6+y[0])
	
	return np.array([g0,g1,g2,g3])
	
# the calculation

ans1_2 = odeint(with_spring, y1, time)
ans2_2 = odeint(with_spring, y2, time)
ans3_2 = odeint(with_spring, y3, time)
ans4_2 = odeint(with_spring, y4, time)
ans5_2 = odeint(with_spring, y5, time)
ans6_2 = odeint(with_spring2, y3, time)
TH1_2 = ans1_2[:,2]
TH2_2 = ans2_2[:,2]
TH3_2 = ans3_2[:,2]
TH4_2 = ans4_2[:,2]
TH5_2 = ans5_2[:,2]
OM1_2 = ans1_2[:,3]
OM2_2 = ans2_2[:,3]
OM3_2 = ans3_2[:,3]
OM4_2 = ans4_2[:,3]
OM5_2 = ans5_2[:,3]
dL1 = ans1_2[:,1]
dL2 = ans2_2[:,1]
dL3 = ans3_2[:,1]
dL4 = ans4_2[:,1]
dL5 = ans5_2[:,1]
delL1 = ans1_2[:,0]
delL2 = ans2_2[:,0]
delL3 = ans3_2[:,0]
delL4 = ans4_2[:,0]
delL5 = ans5_2[:,0]

# graph the results

xpos3_2 = (L_0_3 + ans3_2[:,0])*sin(ans3_2[:,2])
xpos6_2 = (L_0_6 + ans6_2[:,0])*sin(ans6_2[:,2])
ypos3_2 = -(L_0_3 + ans3_2[:,0])*cos(ans3_2[:,2])
ypos6_2 = -(L_0_3 + ans6_2[:,0])*cos(ans6_2[:,2])

#graph trajectory
plt.plot(xpos3_2,ypos3_2, 'm-')
plt.xlabel("X-Position")
plt.ylabel("Y-Position")
plt.title("Trajectory of Springy Pendulum, k=100")
plt.show()

#graph the energy
T = 0.5*m*(L_0_3**2)*ans3_2[:,3]**2
U = m*g*L_0_3*ypos3_2 - 0.5*k*(ans3_2[:,0]**2 - L_0_3)**2
Lagrangian = T - U
Hamiltonian = T + U
#graph the total energy as a function of time
#print(time)
plt.plot(time,Hamiltonian)
plt.xlabel("Time")
plt.ylabel("Total Energy")
#plt.ylim(-5,5)
plt.title("Total Energy vs. Time (With Spring)")
plt.show()

#graph the chaos of the system
plt.plot(time,abs((xpos3_2**2+ypos3_2**2)**2-(xpos6_2**2+ypos6_2**2)**2), 'c-',lw = 1)
plt.xlabel("Time")
plt.ylabel("Difference in Length")
plt.title("Dynamical Chaos vs. Time of Springy Pendulum, k=1.5")
plt.show()

#graph phase plot of theta, omega
plt.plot(TH1_2,OM1_2, 'c-',lw = 1)
plt.plot(TH2_2,OM2_2, 'm-',lw = 1)
plt.plot(TH3_2,OM3_2, 'b-',lw = 1)
plt.plot(TH4_2,OM4_2, 'g-',lw = 1)
plt.plot(TH5_2,OM5_2, 'y-',lw = 1)
plt.xlabel("Theta")
plt.ylabel("Omega")
plt.title("Phase Plot: Theta vs. Omega of Springy Pendulum")
plt.show()

#graph phase plot of L_0, deltaL
plt.plot(delL1,dL1, 'c-',lw = 1)
plt.plot(delL2,dL2, 'm-',lw = 1)
plt.plot(delL3,dL3, 'b-',lw = 1)
plt.plot(delL4,dL4, 'g-',lw = 1)
plt.plot(delL5,dL5, 'y-',lw = 1)
plt.xlabel("L")
plt.ylabel("Ldot")
plt.title("Phase Plot: L vs. Ldot of Springy Pendulum")
plt.show()



#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------



# with spring

# set initial state
y1 = np.array([deltaL,v_0,theta_0,omega_0])
y2 = np.array([deltaL,v_0,theta_0_2,omega_0])
y3 = np.array([deltaL,v_0,theta_0_3,omega_0])
y4 = np.array([deltaL,v_0,theta_0_4,omega_0])
y5 = np.array([deltaL,v_0,theta_0_5,omega_0])

def damped_with_spring(y,time):
	
	g0 = y[1]
	g1 = (L_0_3+y[0])*y[3]*y[3] - k/m*y[0] + g*cos(y[2])
	g2 = y[3]
	g3 = -(g*sin(y[2]) + 2.0*y[1]*y[3])/(L_0_3+y[0]) - 0.317*y[3]
	
	return np.array([g0,g1,g2,g3])

def damped_with_spring2(y,time):
	
	g0 = y[1]
	g1 = (L_0_6+y[0])*y[3]*y[3] - k/m*y[0] + g*cos(y[2])
	g2 = y[3]
	g3 = -(g*sin(y[2]) + 2.0*y[1]*y[3])/(L_0_6+y[0]) - 0.317*y[3]
	
	return np.array([g0,g1,g2,g3])
	
# the calculation

ans1_3 = odeint(damped_with_spring, y1, time)
ans2_3 = odeint(damped_with_spring, y2, time)
ans3_3 = odeint(damped_with_spring, y3, time)
ans4_3 = odeint(damped_with_spring, y4, time)
ans5_3 = odeint(damped_with_spring, y5, time)
ans6_3 = odeint(damped_with_spring2, y3, time)
TH1_3 = ans1_3[:,2]
TH2_3 = ans2_3[:,2]
TH3_3 = ans3_3[:,2]
TH4_3 = ans4_3[:,2]
TH5_3 = ans5_3[:,2]
OM1_3 = ans1_3[:,3]
OM2_3 = ans2_3[:,3]
OM3_3 = ans3_3[:,3]
OM4_3 = ans4_3[:,3]
OM5_3 = ans5_3[:,3]
dL1_3 = ans1_3[:,1]
dL2_3 = ans2_3[:,1]
dL3_3 = ans3_3[:,1]
dL4_3 = ans4_3[:,1]
dL5_3 = ans5_3[:,1]
delL1_3 = ans1_3[:,0]
delL2_3 = ans2_3[:,0]
delL3_3 = ans3_3[:,0]
delL4_3 = ans4_3[:,0]
delL5_3 = ans5_3[:,0]

# graph the results

xpos3_3 = (L_0_3 + ans3_3[:,0])*sin(ans3_3[:,2])
xpos6_3 = (L_0_6 + ans6_3[:,0])*sin(ans6_3[:,2])
ypos3_3 = -(L_0_3 + ans3_3[:,0])*cos(ans3_3[:,2])
ypos6_3 = -(L_0_3 + ans6_3[:,0])*cos(ans6_3[:,2])

#graph trajectory
plt.plot(xpos3_3,ypos3_3, 'm-')
plt.xlabel("X-Position")
plt.ylabel("Y-Position")
plt.title("Trajectory of Damped Springy Pendulum, k=100")
plt.show()

#graph the energy
T = 0.5*m*(L_0_3**2)*ans3_3[:,3]**2
U = m*g*L_0_3*ypos3_3 - 0.5*k*(ans3_3[:,0]**2 - L_0_3)**2
Lagrangian = T - U
Hamiltonian = T + U
#graph the total energy as a function of time
#print(time)
plt.plot(time,Hamiltonian)
plt.xlabel("Time")
plt.ylabel("Total Energy")
#plt.ylim(-5,5)
plt.title("Total Energy vs. Time (With Damped + Spring)")
plt.show()

#graph the chaos of the system
plt.plot(time,abs((xpos3_3**2+ypos3_3**2)**2-(xpos6_3**2+ypos6_3**2)**2), 'c-',lw = 1)
plt.xlabel("Time")
plt.ylabel("Difference in Length")
plt.title("Dynamical Chaos vs. Time of Damped Springy Pendulum, k=1.5")
plt.show()

#graph phase plot of theta, omega
plt.plot(TH1_3,OM1_3, 'c-',lw = 1)
plt.plot(TH2_3,OM2_3, 'm-',lw = 1)
plt.plot(TH3_3,OM3_3, 'b-',lw = 1)
plt.plot(TH4_3,OM4_3, 'g-',lw = 1)
plt.plot(TH5_3,OM5_3, 'y-',lw = 1)
plt.xlabel("Theta")
plt.ylabel("Omega")
plt.title("Phase Plot: Theta vs. Omega of Damped Springy Pendulum")
plt.show()

#graph phase plot of L_0, deltaL
plt.plot(delL1_3,dL1_3, 'c-',lw = 1)
plt.plot(delL2_3,dL2_3, 'm-',lw = 1)
plt.plot(delL3_3,dL3_3, 'b-',lw = 1)
plt.plot(delL4_3,dL4_3, 'g-',lw = 1)
plt.plot(delL5_3,dL5_3, 'y-',lw = 1)
plt.xlabel("L")
plt.ylabel("Ldot")
plt.title("Phase Plot: L vs. Ldot of Damped Springy Pendulum")
plt.show()


