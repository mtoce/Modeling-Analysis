# Michael Toce
# Modelling Analysis
# Assignment 2: Linear Programming

# write a mimization optimization program for an ironman training diet
# variables: energy, fats, carbs, proteins, calcium, iron, sodium, chloroform

# write a minimization optimization program for finding times when to train for ironman / smash, while maximizing the amount of time I get to play smash

#tableau for a traithlete:
#500g/day carbs , 100g/day protein, 100g/day fat, <2500 calories/day
# try to make it so i can eat more calories on the weekend cause i have longer workouts

# ask the user what he/she wants to minimize

import numpy as np
import scipy.optimize as sp
import astropy.table as ap
import matplotlib.pyplot as plt
# part A
# food = []
calories = []

food = np.loadtxt('groceries1.txt', usecols=(2,3,4,5,6), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries1.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
calories = np.loadtxt('groceries1.txt', usecols=(1,),skiprows=2, unpack=True)
c1 = calories
f = np.array([1,1,1,0.001,0.001])
food = (food.T*f).T
food = 0.01*food
b1 = [-70., -310., -50., -1., -.018, 2000.]
food = np.vstack((food, np.ones(len(food[0]))))
A1 = food

res1 = sp.linprog(c1, A_ub=A1, b_ub=b1)
print(res1)
output1 = np.array([0.        ,     0.        ,   477.42464123,     0.        ,     0.        ,     0.        ,     0.        ,   291.75012793,
0.        ,     0.        ,     0.        ,     0.        ,
0.        ,     0.        ,     0.        ,  1044.66919807,
0.        ,     0.        ,   186.15603277,     0.        ,
0.        ,     0.        ,     0.])

calA1 = (477.42464123/100)*c1[2] + (291.75012793/100)*c1[7] + (1044.66919807/100)*c1[15] + (186.15603277/100)*c1[18]
print("cal part A (original): ", calA1)
fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output1, width, color = 'purple')
ax.set_ylabel('grams/day')
ax.set_title('Part A: Minimize Calories With Original List')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()

# change to McDonalds french fries

food = np.loadtxt('groceries3.txt', usecols=(2,3,4,5,6), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries3.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
calories = np.loadtxt('groceries3.txt', usecols=(1,),skiprows=2, unpack=True)
c2 = calories
f = np.array([1,1,1,0.001,0.001])
food = (food.T*f).T
food = 0.01*food
b2 = [-70., -310., -50., -1., -.018, 2000.]
food = np.vstack((food, np.ones(len(food[0]))))
A2 = food

res2 = sp.linprog(c2, A_ub=A2, b_ub=b2)
print(res2)

output2 = np.array([   0.        ,    0.        ,    0.        ,    0.        ,          0.        ,    0.        ,   39.35059641,    0.        ,          0.        ,   41.63745924,    0.        , 0.        ,          0.        ,    0.        ,    0.        ,  831.39422306,          0.        ,    0.        ,  617.46013857,   0.        ,          0.        ,  333.76203542,  136.3955473 ])
totalcal2 = (39.35/100)*c2[6] + (41.64/100)*c2[9] + (831.40/100)*c2[15] + (617.46/100)*c2[18] + (333.76/100)*c2[21] + (136.40/100)*c2[22]
calswitchff = (477.42/100)*c2[2] + (291.75/100)*c1[7] + (1044.67/100)*c1[15] + (186.16/100)*c1[18]
print("cal switch fries: ", calswitchff)
print("cal part A (ff): ", totalcal2)
fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output2, width, color = 'y')
ax.set_ylabel('grams/day')
ax.set_title('Part A: Change to McDonald\'s Fries')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()


# part B

food = np.loadtxt('groceries1.txt', usecols=(1,3,4,5,6), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries1.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
fats = np.loadtxt('groceries1.txt', usecols=(2,),skiprows=2, unpack=True)
c3 = fats
f = np.array([1,1,1,0.001,0.001])
food = (food.T*f).T
food = 0.01*food
b3 = [-2000, -310., -50., -1., -.018, 2000.]
food = np.vstack((food, np.ones(len(food[0]))))
A3 = food

res3 = sp.linprog(c3, A_ub=A3, b_ub=b3)
print(res3)
output3 = np.array([   0.        ,    0.        ,    0.        ,    0.        ,          0.        ,    0.        ,    0.        ,  222.62443439,          0.        ,    0.        ,    0.        ,    0.        ,          0.        ,  961.4479638 ,    0.        ,  815.92760181,          0.        ,    0.        ,    0.        ,    0.        ,          0.        ,    0.        ,    0.        ])

fatB = (222.62/100)*c3[7] + (961.45/100)*c3[13] + (815.93/100)*c3[15]
calB = (222.62/100)*c1[7] + (961.45/100)*c1[13] + (815.93/100)*c1[15]
calBnoround = (222.62443439/100)*c1[7] + (961.4479638/100)*c1[13] + (815.92760181/100)*c1[15]
print("fat part B: ", fatB)
print("cal part B: ", calB)
print("cal part B no round: ", calBnoround)
fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output3, width, color = 'orange')
ax.set_ylabel('grams/day')
ax.set_title('Part B: Minimize Fats')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()

# part C

food = np.loadtxt('groceries2.txt', usecols=(2,3,4,5,6,7,9,10,12,13,14), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries2.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
calories = np.loadtxt('groceries2.txt', usecols=(1,),skiprows=2, unpack=True)
c4 = calories
f = np.array([1,1,1,.001,.001,.001,1,1,.001,.000001,.001])
food = (food.T*f).T
food = 0.01*food
b4 = [-70, -310., -50., -1., -.018, -2.4, -25, -25,  -.0013, -.0000024, -.085, 2000.]
#     fat  carb  prot   Ca    Fe     Na  fiber sug     vB6      vB12      vC    cal min
#.000006,.000000025
#-.0159,-.000015,
# va      vD

food = np.vstack((food, np.ones(len(food[0]))))
A4 = food

res4 = sp.linprog(c4, A_ub=A4, b_ub=b4)
print(res4)
output4 = np.array([48.77546168,  572.87695122,    0.        ,    0.        ,          0.        ,    0.        ,    0.        ,  358.34651258,          0.        ,   63.26664774,    0.        ,    0.        ,          0.        ,    0.        ,  786.39626976,    0.        ,          0.        ,   39.54081261,    0.        ,    0.        ,    0.        ])
calpartC = (48.78/100)*c4[0] + (572.88/100)*c4[1] + (358.35/100)*c4[7] + (63.27/100)*c4[9] + (786.40/100)*c4[14] + (39.54/100)*c4[17]
print("cal part C: ", calpartC)

fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output4, width, color = 'b')
ax.set_ylabel('grams/day')
ax.set_title('Part C: Added Constraints')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()

#part C failure with vitamin A & D

food = np.loadtxt('groceries2.txt', usecols=(2,3,4,5,6,7,9,10,11,12,13,14,15), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries2.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
calories = np.loadtxt('groceries2.txt', usecols=(1,),skiprows=2, unpack=True)
c5 = calories
f = np.array([1,1,1,.001,.001,.001,1,1,.000006,.001,.000001,.001,.000000025])
food = (food.T*f).T
food = 0.01*food
b5 = [-70, -310., -50., -1., -.018, -2.4, -25, -25, -.0159,-.0013, -.0000024, -.085, -.0000015, 2000.]
#     fat  carb  prot   Ca    Fe     Na  fiber sug     vB6      vB12      vC    cal min
#.000006,.000000025
#-.0159,-.000015,
# va      vD

food = np.vstack((food, np.ones(len(food[0]))))
A5 = food

res5 = sp.linprog(c5, A_ub=A5, b_ub=b5)
print(res5)
output5 = np.array([  29.45802137,  766.86588876,    0.        ,    0.        ,          0.        ,    0.        ,   14.86042318,  342.74071453,          0.        ,   48.41659857,    0.        ,    0.        ,          0.        ,    0.        ,  551.70618846,    0.        ,          0.        ,   40.62206415,   63.98724598,    0.        ,    0.        ])
calvitaminsC = (29.46/100)*c5[0] + (766.87/100)*c5[1] + (14.86/100)*c5[6] + (342.74/100)*c5[7] + (48.42/100)*c5[9] + (551.71/100)*c5[14] + (40.62/100)*c5[17] + (63.99/100)*c5[18]
print("cal part C w/ vitamins: ", calvitaminsC)

fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output5, width, color = 'b')
ax.set_ylabel('grams/day')
ax.set_title('Part C: Added Constraints With Vitamins A&D')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()

# Part D

food = np.loadtxt('groceries5.txt', usecols=(2,3,4,5,6), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries5.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
price = np.loadtxt('groceries5.txt', usecols=(15,),skiprows=2, unpack=True)
c6 = price
f = np.array([1,1,1,0.001,0.001])
food = (food.T*f).T
food = 0.01*food
b6 = [-70., -310., -50., -1., -.018, 2000.]
food = np.vstack((food, np.ones(len(food[0]))))
A6 = food

res6 = sp.linprog(c6, A_ub=A6, b_ub=b6)
print(res6)
output6 = np.array([   0.        ,    0.        ,    0.        ,    0.        ,          0.        ,   60.70181666,    0.        ,  646.64990037,          0.        ,   54.85036936,    0.        ,    0.        ,          0.        ,    0.        ,    0.        ,    0.        ,          0.        ,    0.        ,    0.        ,    0.        ,    0.        ])
priceD = (60.70/100)*c6[5] + (646.65/100)*c6[7] + (54.85/100)*c6[9]
calD = (60.70/100)*c5[5] + (646.65/100)*c5[7] + (54.85/100)*c5[9]
print("cal part D (original): ", calD)
print("price part D (original): ", priceD)

fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output6, width, color = 'r')
ax.set_ylabel('grams/day')
ax.set_title('Part D: Minimize Price with Original Constraints')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()



# part D with added constraints

food = np.loadtxt('groceries5.txt', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14), skiprows=2, unpack=True)*(-1)
foodnames = np.loadtxt('groceries5.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
calories = np.loadtxt('groceries5.txt', usecols=(15,),skiprows=2, unpack=True)
c7 = calories
f = np.array([1,1,1,.001,.001,.001,1,1,.000006,.001,.000001,.001,.000000025])
food = (food.T*f).T
food = 0.01*food
b7 = [-70, -310., -50., -1., -.018, -2.4, -25, -25, -.0159,-.0013, -.0000024, -.085, -.0000015, 2000.]
#     fat  carb  prot   Ca    Fe     Na  fiber sug     vB6      vB12      vC    cal min
#.000006,.000000025
#-.0159,-.000015,
# va      vD

food = np.vstack((food, np.ones(len(food[0]))))
A7 = food

res7 = sp.linprog(c7, A_ub=A7, b_ub=b7)
print(res7)
output7 = np.array([  67.40123554,    0.        ,    0.        ,    0.        ,          0.        ,  136.34699098,    0.        ,  520.03939598,          0.        ,   39.25414186,    0.        ,   12.17059132,          0.        ,    0.        ,    0.        ,    0.        ,          0.        ,  132.27848596,   66.56386402,    0.        ,    0.        ])
priceD2 = (67.40/100)*c7[0] + (136.37/100)*c7[5] + (520.04/100)*c7[7] + (39.25/100)*c7[9] + (12.17/100)*c7[11] + (132.28/100)*c7[17] + (66.56/100)*c7[18]
calD2 = (67.40/100)*c5[0] + (136.37/100)*c5[5] + (520.04/100)*c5[7] + (39.25/100)*c5[9] + (12.17/100)*c5[11] + (132.28/100)*c5[17] + (66.56/100)*c5[18]
priceC = (29.46/100)*c7[0] + (766.87/100)*c7[1] + (14.86/100)*c7[6] + (342.74/100)*c7[7] + (48.42/100)*c7[9] + (551.71/100)*c7[14] + (40.62/100)*c7[17] + (63.99/100)*c7[18]
print("cal part D added constraints: ", calD2)
print("price part D  added constraints: ", priceD2)
print("price part C added constraints: ", priceC)
fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnames))
data = ax.bar(ind, output7, width, color = 'r')
ax.set_ylabel('grams/day')
ax.set_title('Part D: Minimize Price with Added Constraints')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnames)
plt.setp(labels, rotation=30 )
plt.show()

# part E triathletes

food = np.loadtxt('groceries6.txt', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14), skiprows=2, unpack=True)*(-1)
foodnamesnew = np.loadtxt('groceries6.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
price = np.loadtxt('groceries6.txt', usecols=(15,),skiprows=2, unpack=True)
calories = np.loadtxt('groceries6.txt', usecols=(1,),skiprows=2, unpack=True)
c8 = price
c9 = calories
f = np.array([1,1,1,1,.001,.001,.001,1,1,.000006,.001,.000001,.001,.000000025])
food = (food.T*f).T
food = 0.01*food
b8 = [-2500,-70, -500., -100., -1., -.018, -2.4, -25, -25, -.0159,-.0013, -.0000024, -.085, -.0000015, 2000.]
#     fat  carb  prot   Ca    Fe     Na  fiber sug     vB6      vB12      vC    cal min

food = np.vstack((food, np.ones(len(food[0]))))
A8 = food

res8 = sp.linprog(c8, A_ub=A8, b_ub=b8)
print(res8)
output8 = np.array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,         0.00000000e+00,   3.65279566e+02,   0.00000000e+00,
         6.12202724e+02,   0.00000000e+00,   2.77115823e+01,         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,         0.00000000e+00,   6.89655172e+01,   0.00000000e+00,         0.00000000e+00,   8.34618019e+02,   4.66426247e+00,         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,         0.00000000e+00,   3.11521603e-01,   0.00000000e+00])

cal8 = (365.28/100)*c9[4] + (612.20/100)*c9[6] + (23.71/100)*c9[8] + (68.96/100)*c9[16] + (834.62/100)*c9[19] + (4.66/100)*c9[20] + (.31/100)*c9[25]
price8 = (365.28/100)*c8[4] + (612.20/100)*c8[6] + (23.71/100)*c8[8] + (68.96/100)*c8[16] + (834.62/100)*c8[19] + (4.66/100)*c8[20] + (.31/100)*c8[25]
print(cal8)
print(price8)

fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnamesnew))
data = ax.bar(ind, output8, width, color = 'g')
ax.set_ylabel('grams/day')
ax.set_title('Triathlete Training Minimizing Price')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnamesnew)
plt.setp(labels, rotation=30 )
plt.show()

# minimize calories now (triathletes)

food = np.loadtxt('groceries6.txt', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14), skiprows=2, unpack=True)*(-1)
foodnamesnew = np.loadtxt('groceries6.txt', usecols=(0,), skiprows=2, unpack=True, dtype = str)
price = np.loadtxt('groceries6.txt', usecols=(15,),skiprows=2, unpack=True)
calories = np.loadtxt('groceries6.txt', usecols=(1,),skiprows=2, unpack=True)
c8 = price
c9 = calories
f = np.array([1,1,1,.001,.001,.001,1,1,.000006,.001,.000001,.001,.000000025])
food = (food.T*f).T
food = 0.01*food
b9 = [-70, -500., -100., -1., -.018, -2.4, -25, -25, -.0159,-.0013, -.0000024, -.085, -.0000015, 2000.]
#     fat  carb  prot   Ca    Fe     Na  fiber sug     vB6      vB12      vC    cal min

food = np.vstack((food, np.ones(len(food[0]))))
A8 = food

res9 = sp.linprog(c9, A_ub=A8, b_ub=b9)
print(res9)
#output9 = np.array([  81.79323786,    0.        ,    0.        ,    0.        ,          0.        ,   89.68815011,   70.57108877,    0.        ,         29.42324279,    0.        ,    0.        ,    0.        ,          0.        ,    0.        ,    0.        ,    0.        ,         42.34038517,  759.56900564,    0.        ,  920.89939116,          1.0764611 ,    0.        ,    0.        ,    0.        ,          4.6390374 ,    0.        ,    0.        ])

cal9 = (81.79/100)*c9[0] + (89.69/100)*c9[5] + (70.57/100)*c9[6] + (29.42/100)*c9[8] + (42.34/100)*c9[16] + (759.57/100)*c9[17] + (920.90/100)*c9[19] + (1.0765/100)*c9[20] + (4.639/100)*c9[24]
price9 = (81.79/100)*c8[0] + (89.69/100)*c8[5] + (70.57/100)*c8[6] + (29.42/100)*c8[8] + (42.34/100)*c8[16] + (759.57/100)*c8[17] + (920.90/100)*c8[19] + (1.0765/100)*c8[20] + (4.639/100)*c8[24]
print("cal9: ",cal9)
print("price9: ",price9)

fig, ax = plt.subplots()
width = 0.5
ind = np.arange(len(foodnamesnew))
data = ax.bar(ind, output9, width, color = 'g')
ax.set_ylabel('grams/day')
ax.set_title('Triathlete Training Minimizing Calories')
ax.set_xticks(ind)
labels = ax.set_xticklabels(foodnamesnew)
plt.setp(labels, rotation=30 )
plt.show()

'''
sex = raw_input("Are you a male or female? (Type m or f)")
age = raw_input("Are you young? (15-30) prime? (31-45) or older? (46-90)")
dist = raw_input("What distance ironman are you doing? (half, full)")


# young male sprint
if age == young and sex == m and dist == half:
	
if age == young and sex == m and dist == full:
if age == young and sex == f and dist == half:
if age == young and sex == f and dist == full:
if age == prime and sex == m and dist == half:
if age == prime and sex == m and dist == full:
if age == prime and sex == f and dist == half:
if age == prime and sex == f and dist == full:
if age == older and sex == m and dist == half:
if age == older and sex == m and dist == full:
if age == older and sex == f and dist == half:
if age == older and sex == f and dist == full:
else:
	print("Whoops! Looks like you inputted incorrectly, try again!")
'''
