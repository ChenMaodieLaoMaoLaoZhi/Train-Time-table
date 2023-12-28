# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 02:04:51 2023

@author: 陈纪程
"""

from coptpy import *
import pandas as pd
import re
import matplotlib.pyplot as plt

def get_variance(records):
    average = sum(records) / len(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)

def see_solution():
    print("Variable solution:")
    for var in all_vars:
        if re.match('[A,D]', var.name):
            print("{0}:{1}".format(var.name, var.x))
            
def get_max(records):
    maxx = 0
    for each in records:
        if each >= maxx:
            maxx = each
    return maxx

env = Envr()

model = env.createModel("Train Timetabling Problem")

S = [[1,0,0,0,0,1,1],[1,1,1,0,1,0,1],
     [1,1,0,1,0,1,1],[1,1,0,0,1,0,1],
     [1,1,0,0,0,1,1],[1,1,0,0,1,0,1],
     [1,1,1,1,1,1,1],[1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1],[1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1],[1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1],[1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1],[1,1,1,1,1,1,1]]
R = [[10,9],[20,18],[14,12],[8,7],[8,7],[10,8]]

Stay_Sche = pd.DataFrame(S)
Stay_Sche['Speed_type'] = [1 for i in range(6)] + [0 for i in range(10)]

Run_through = pd.DataFrame(R)

Time_between = []
for i in range(16):
    Time_between.append([])
    for station in range(6):
        Time_between[i].append(2*Stay_Sche[station][i]\
              + Run_through[Stay_Sche['Speed_type'][i]][station] + 2*Stay_Sche[station+1][i])

A = model.addVars(16, 7, vtype=COPT.INTEGER, nameprefix="A")
D = model.addVars(16, 7, vtype=COPT.INTEGER, nameprefix="D")

# no overtake
M = 150
x = model.addVars(16, 16, 6, 2, vtype=COPT.BINARY, nameprefix='x')
for i in range(16):
    for j in range(16):
        if j != i:
            for station in range(6):
                model.addConstr(1 <= 1 + D[(i,station)] - D[(j,station)] + M*x[(i,j,station,0)])
                model.addConstr(1 + D[(i,station)] - D[(j,station)] + M*x[(i,j,station,0)] <= M)
                model.addConstr(1 <= 1 + A[(i,station+1)] - A[(j,station+1)] + M*x[(i,j,station,1)])
                model.addConstr(1 + A[(i,station+1)] - A[(j,station+1)] + M*x[(i,j,station,1)] <= M)
                model.addConstr(x[(i,j,station,1)] >= x[(i,j,station,0)])

# headway
hw = 3
y = model.addVars(16, 16, 7, 4, vtype=COPT.BINARY, nameprefix='y')
z = model.addVars(16, 16, 7, 2, vtype=COPT.BINARY, nameprefix='z')
for i in range(16):
    for j in range(16):
        if j != i:
            for station in range(7):  
                model.addConstr(hw + A[(i,station)] - A[(j,station)] <= M*z[(i,j,station,0)])
                model.addConstr(hw + A[(j,station)] - A[(i,station)] <= M*(1-z[(i,j,station,0)]))
                model.addConstr(hw + D[(i,station)] - D[(j,station)] <= M*z[(i,j,station,1)])
                model.addConstr(hw + D[(j,station)] - D[(i,station)] <= M*(1-z[(i,j,station,1)]))

                model.addConstr(1 <= 1 + A[(i,station)] - A[(j,station)] + M*y[(i,j,station,0)])
                model.addConstr(1 + A[(i,station)] - A[(j,station)] + M*y[(i,j,station,0)] <= M)
                model.addConstr(1 <= A[(i,station)] + 3 - A[(j,station)] + M*y[(i,j,station,1)])
                model.addConstr(A[(i,station)] + 3 - A[(j,station)] + M*y[(i,j,station,1)] <= M)
                model.addConstr(y[(i,j,station,1)] >= y[(i,j,station,0)])

                model.addConstr(1 <= 1 + D[(i,station)] - D[(j,station)] + M*y[(i,j,station,2)])
                model.addConstr(1 + D[(i,station)] - D[(j,station)] + M*y[(i,j,station,2)] <= M)
                model.addConstr(1 <= D[(i,station)] + 3 - D[(j,station)] + M*y[(i,j,station,3)])
                model.addConstr(D[(i,station)] + 3 - D[(j,station)] + M*y[(i,j,station,3)] <= M)
                model.addConstr(y[(i,j,station,3)] >= y[(i,j,station,2)])

# Time Resource
model.addConstrs(D[(i,6)] <= 150 for i in range(16))

# Waiting Time
for i in range(16):
    for station in range(7):
        if Stay_Sche[station][i] == 1 and (station != 0 and station != 6):
            model.addConstr(2 <= D[(i,station)] - A[(i,station)])
            model.addConstr(D[(i,station)] - A[(i,station)] <= 15)
        else:
            model.addConstr(D[(i,station)] == A[(i,station)])
      
# Time between station
for i in range(16):
    for station in range(6):
        model.addConstr(A[(i,station+1)] - D[(i,station)] >= Time_between[i][station])

model.setObjective(sum([(D[(i,6)] - A[(i,0)]) for i in range(16)]), COPT.MINIMIZE)

model.setParam(COPT.Param.TimeLimit, 10)

model.solve()

print("Objective value: {}".format(model.objval))
all_vars = model.getVars() 
points_x = [[0 for j in range(14)] for i in range(16)]
for var in all_vars:
    for i in range(16):
        for station in range(7):
            if re.match('A.{0}.{1}.'.format(i,station), var.name):
                points_x[i][2*station] = var.x
            if re.match('D.{0}.{1}.'.format(i,station), var.name):
                points_x[i][2*station+1] = var.x
                    
points_y = [0,0,1,1,2,2,3,3,4,4,5,5,6,6]

plt.figure(figsize=(6,4),dpi=240)
for i in range(16):
    plt.plot(points_x[i], points_y,label=i+1)
plt.legend()
plt.show()

lines = ['G{0}'.format(i) for i in range(1,32,2)]
time_table = pd.DataFrame(points_x,index=lines,
                          columns=['A0','A1','B0','B1','C0','C1','D0','D1','E0','E1','F0','F1','G0','G1'])
time_table.to_excel('D:/跟课系列/曾经是优化理论的文件夹/优化II/Final Project 火车排班问题/time_table.xlsx')

