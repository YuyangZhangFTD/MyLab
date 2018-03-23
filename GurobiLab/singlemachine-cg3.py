# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 23:21:37 2017

@author: frankWXD
"""
# Single Machine Schdule, Column Generation
# 指定job数N,随机生成p[j],c[j][t],自动生成x[k][j][t]
# 求解出的是 Original Formulation的松弛问题,自变量x[j][t]的上界值???

import time
import numpy as np
from gurobipy import *
# 设置参数（定义常量）：考虑 N个job, T个时段的单机调度问题
N = 10 # 指定单机调度的job数
p = [] # 每个job的加工时间p[j]
for jj in range(N):  # 保证每个job的加工时间在1-3之间
    p.append( 1 + (np.random.randint(100))%5 )    
T = sum(p) + 1 # 时段数T至少是所有job的加工时间之和+1
c = np.zeros((N,T)) # c[j][t] job j在t时段开始加工的成本
for jj in range(N):
    #for tt in range(T-p[jj]+1):
    for tt in range(T):
        c[jj][tt] = 1 + (np.random.randint(100))%8 # 保证成本在1-8之间

# 用多面体P的K个顶点，表达多面体内部的任意一点
K = 2 # 初始可行解个数：K=2
x = [[[0 for t in range(T)] for j in range(N)] for k in range(K)]
# ================
# 每个job单独做的解,是无效的
#for kk in range(N):
#    x[kk][kk][0] = 1 # 第k个解是：第k个job在时段1就开始加工，其他job不开始
# (1)所有job顺次做完的解    
for jj in range(N):
    if jj == 0:
        x[0][jj][0]=1
    else:
        sum1 = 0
        for j in range(jj):
            sum1 += p[j] 
        x[0][jj][0+sum1] = 1
# N=5时枚举的例子(1)
#x[0][0][0]=1
#x[0][1][0+p[0]]=1
#x[0][2][0+p[0]+p[1]]=1
#x[0][3][0+p[0]+p[1]+p[2]]=1
#x[0][4][0+p[0]+p[1]+p[2]+p[3]]=1

# (2)所有job逆次做完的解
for jj in range(N-1,-1,-1):
    if jj == N-1:
        x[1][jj][0]=1
    else:
        sum1 = 0
        for j in range(N-1,jj,-1):
            sum1 += p[j] 
        x[1][jj][0+sum1] = 1
# N=5时枚举的例子(2)
#x[1][4][0] = 1
#x[1][3][0+p[4]] = 1
#x[1][2][0+p[4]+p[3]] = 1
#x[1][1][0+p[4]+p[3]+p[2]] = 1
#x[1][0][0+p[4]+p[3]+p[2]+p[1]] = 1

# 输出初始解,查看是否合理
# print('Initial Solutions: ')
# for ii in range(6):
#   for jj in range(5):
#       for kk in range(18):
#           if x[ii][jj][kk]>0:
#               print('x[%d][%d][%d] = %d'%(ii,jj,kk,x[ii][jj][kk]))
# =================

MAX_CGTIMES =1000  # 最大列生成的迭代次数
CGtime = 0 # 列生成loop计算时间
def reportRMP(model):  # 报告Restrict Master Problem限制主问题
    if model.status == GRB.OPTIMAL:  # 如果主问题求得了最优，
        print("RMP_Total Cost: ", model.objVal)  # 输出的目标值是共用几个整料        
        # 获取RMP中的变量
        # var = model.getVars() 
        # for i in range(model.numVars): # 遍历所有变量，输出每个变量的值
        #     print(var[i].varName, " = ", var[i].x)
        # print("\n")  
        # # 获取主问题中的约束
        # con = model.getConstrs() 
        # for i in range(model.numConstrs): #输出每个约束对应的pi：对偶值/影子价格;所有pi值构成行向量PI
        #     print(con[i].constrName, " = ", con[i].pi)
        # print("\n")
        # con[i].pi: Dual values for the computed solution
        # (also known as shadow prices). 
        # This array contains one entry for each row of A.
def reportSUB(model): # 报告Princing Problem,即Sub-Problem 子问题
    if model.status == GRB.OPTIMAL: # 如果子问题求得了最优，
        print("SUB_Ck: ", model.objVal, "\n") # 输出目标值，这里应该是C_k:resuced cost
        # 判断是否最优
        # if model.objVal <= 1e-6:  # 如果子问题的目标值 <= 0,考虑deltaZ为加号情况
        #     var = model.getVars()  # 获取子问题的变量(此时不是最优，可以继续小)
        #     # 遍历每个子问题变量
        #     for i in range(model.numVars): 
        #         print(var[i].varName, " = ", var[i].x) 
        #     print("\n") # 输出每个变量名与变量值
def reportMIP(model): # 报告整个MIP问题
    if model.status == GRB.OPTIMAL: # 如果求得了最优，输出目标：总成本
        print(" RMP best solution: ", model.objVal)
        # 获取每个变量，输出它们的值
        # var = model.getVars() 
        # for i in range(model.numVars):
        #     if var[i].x != 0:
        #         print(var[i].varName, " = ", var[i].x)

# 列生成开始计时
Tstart1 = time.clock()

# 构造RMP和SUB
rmp = Model("rmp") # rmp 限制主问题模型
sub = Model("sub") # sub 子问题模型
rmp.setParam("OutputFlag", 0) # 主问题与子问题均不输出求解详细信息
sub.setParam("OutputFlag", 0)
# construct RMP 构造主问题   
# 变量
Lambda = [0 for k in range(K)]
for kk in range(K):
    Lambda[kk] = rmp.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Lambda[%d]'%(kk))
# 目标
obj = LinExpr()
for kk in range(K):
    for jj in range(N):
        for tt in range(T-p[jj]+1):
            obj += c[jj][tt]*x[kk][jj][tt]*Lambda[kk]
rmp.setObjective(obj,GRB.MINIMIZE)
# 约束  
rmp_con = [] # 包含主问题约束的list
for jj in range(N):
    sum1 = LinExpr()
    for kk in range(K):
        for tt in range(T-p[jj]+1):
            sum1 += x[kk][jj][tt]*Lambda[kk]
    rmp_con.append(rmp.addConstr(sum1 == 1,'Constraint(1)_j=%d'%(jj)))
sum1 = LinExpr()
for kk in range(K):
    sum1 += Lambda[kk]
rmp_con.append(rmp.addConstr(sum1 == 1,'Constraint(2)'))
# end RMP  
#rmp.update()
#rmp.optimize()
#rmp.write("singlemachine.lp")

# construct SUB 构造子问题
# 变量
sub_x = [[0 for i in range(T)] for j in range(N)]
for jj in range(N):
    for tt in range(T-p[jj]+1):
        sub_x[jj][tt] = sub.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='sub_x[%d][%d]'%(jj,tt))
# 只设置约束，主问题解出pi值再设置目标
for tt in range(T):
    sum1 = LinExpr()
    for jj in range(N):
        for ss in range( tt-p[jj]+1,tt+1 ): # 求和下限不从1开始，则上限+1
            sum1 += sub_x[jj][ss]
    sub.addConstr(sum1 <= 1,'Constraint(3)_t=%d'%(tt))
# end SUB 
# 列生成算法主循环
#print("               *** Column Generation Loop ***               \n")
for count in range(MAX_CGTIMES):
#    print("Iteration: ",count,"\n") # 第count次迭代,int与str不使用加号+连接
    rmp.optimize() # 求解RMP
    CGtime = CGtime + rmp.runtime
#    print( '主问题求解时间:%f'%(rmp.runtime) )
    reportRMP(rmp) # report RMP
    # rmp_pi取出主问题约束组对应的Pi值,注意其中包括了最后一个元素1，对应凸约束
    rmp_pi = rmp.getAttr("Pi", rmp.getConstrs())
    rmp_pi0 = rmp_pi[N] # 取出最后一个约束对应的pi值,即文中的alpha
    rmp_pi = rmp_pi[:N]  # 取出前N个约束对应的pi值
    # 设置子问题目标
    sub_obj = LinExpr()
    for jj in range(N):
        for tt in range(T-p[jj]+1):
            sub_obj += (c[jj][tt] - rmp_pi[jj])*sub_x[jj][tt]
    sub_obj = sub_obj - rmp_pi0  # 【注意】减去alpha这一步构造目标是必不可少的，不能忽略
    sub.setObjective(sub_obj, GRB.MINIMIZE)
    # 求解子问题，并report子问题
    sub.optimize()
    CGtime = CGtime + sub.runtime
#    print( '子问题求解时间:%f'%(sub.runtime) )
    reportSUB(sub)     
    # 如果子问题的目标值>-0.000001,即目标值>0,则达到最优
    if sub.objVal > -1e-3:
        break
    
    # 记录子问题每次迭代后求得的x[k][j][t],添加指定初始值的x中
    tempx = [[0 for t in range(T)] for j in range(N)]   
    for jj in range(N):
        for tt in range(T-p[jj]+1):
            if sub_x[jj][tt].x != 0:
                tempx[jj][tt] = sub_x[jj][tt].x
    x.extend([tempx])
    # 此时，将子问题解出的x[jj][tt],实际上是第 k=count+K 的x[k][j][t]写入x
    
    # 如果当前子问题解出来后判断，当前未达到最优，则取出子问题变量的值处理,准备入基列的系数
    # 取出SUB问题的解,对每个jj对tt求和,形成一列，最后添加一个1,得到待添加到RMP中的系数列
    rmp_coeff = []
    for jj in range(N):
        sum1 = 0
        for tt in range(T-p[jj]+1):
            sum1 += sub_x[jj][tt].x
        rmp_coeff.append( sum1 )
    rmp_coeff.append( 1 )
    # 新变量在目标函数中的系数
    rmp_c_coeff = 0
    for jj in range(N):
        for tt in range(T-p[jj]+1):
            rmp_c_coeff += c[jj][tt]*sub_x[jj][tt].x
    # SUB列系数 + RMP约束组：子问题变量值加入到主问题约束矩阵
    rmp_col = Column(rmp_coeff, rmp_con)
    # 添加系数对应的变量：addVar最后一个参数，将变量添加到刚刚构造的添加列的约束组中
    rmp.addVar(0.0, GRB.INFINITY, rmp_c_coeff, GRB.CONTINUOUS, "cg_%d"%(count), rmp_col) 
#print("               *** End CG Loop ***               \n")    
# 列生成计时结束，输出列生成时间
Tend1 = time.clock()
print( 'Count = %d'%(count))
print( 'CG Time：%f'%((Tend1-Tstart1)))
print( '列生成解松弛问题的求解时间：%f'%(CGtime))


# 列生成过程结果时，证明此时的RMP已经达到最优
# 根据一系列子问题求得的x[k][j][t]和主问题的lambda[k] 计算 x[k][j]>=0
# 即Original Formulation的松弛解

# 前面求解子问题时，已经获取了x[k][j][t]在x中,则计算lambda[k]
LambdaValue = []
LambdaList = rmp.getVars() # 取出RMP中的变量
for each in LambdaList:
    LambdaValue.append(each.x) # 获取Lambda[k]的取值
LambdaK = len(LambdaValue) # RMP中变量个数K被更新

# 根据sum_k{x[k][j][t]*lambda[k]} 计算 x[j][t]
IPrelax_x = [[0 for t in range(T)] for j in range(N)]
for jj in range(N):
    for tt in range(T-p[jj]+1):
        IPrelax_x[jj][tt] =  sum( x[kk][jj][tt]*LambdaValue[kk] for kk in range(LambdaK))
# 输出Original Formulation的松弛解：Relaxation Solution
# 即,得到的是 Original Formulation的上界值
# 如果上界值就是01值,则找到了最优；否则可计算下界值,
# 上界 ==下界则最优,否则上下界之差 = optimilaty gap

# 输出CG产生的松弛解
print("            *** 输出CG产生的松弛解 ***            \n") 
#for jj in range(N):
#    for tt in range(T-p[jj]+1):
#        if IPrelax_x[jj][tt] != 0:
#            print('IPrelax_x[%g][%g] = %g'%(jj,tt,IPrelax_x[jj][tt]))

obj_CG = 0  # 求出目标函数值，并输出
for jj in range(N):
    for tt in range(T-p[jj]+1):
        obj_CG += c[jj][tt]*IPrelax_x[jj][tt]
print( 'Optimal value by Column Generation : %g'%(obj_CG) )




# 对比直接解原模型的Original Formulation的松弛
print("               *** 直接解模型的Original Formulation的松弛 ***               \n") 
# 构建模型
m = Model("SingleMachine")
m.setParam("OutputFlag", 0)
# 变量
x = [[0 for i in range(T)] for j in range(N)]
for jj in range(N):
    for tt in range(T-p[jj]+1):
#        x[jj][tt] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='x[%d][%d]'%(jj,tt))
        x[jj][tt] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x[%d][%d]'%(jj,tt))

# 目标
obj = LinExpr()
for jj in range(N):
    for tt in range(T-p[jj]+1):
        obj += c[jj][tt]*x[jj][tt]
m.setObjective(obj,GRB.MINIMIZE)
# 约束
for jj in range(N):
    sum1 = LinExpr()
    for tt in range(T-p[jj]+1):
        sum1 += x[jj][tt]
    m.addConstr(sum1 == 1,'Constraint(1)_j=%d'%(jj))
for tt in range(T):
    sum1 = LinExpr()
    for jj in range(N):
        for ss in range( tt-p[jj]+1,tt+1 ): # 求和下限不从1开始，则上限+1
            sum1 += x[jj][ss]
    m.addConstr(sum1 <= 1,'Constraint(2)_t=%d'%(tt))
# 求解
m.update()
m.optimize()
print( 'Optimal value in Original Formulation: %g'%m.objVal )
# 输出最优解
#for jj in range(N):
#   for tt in range(T-p[jj]+1):
#       if ((x[jj][tt].x) > 0):
#            print( 'x[%d][%d] = %g'%(jj,tt,x[jj][tt].x) )
# 输出运行时间
print('直接解原模型的松弛,求解时间：%f'%(m.runtime))

##########################################################################


# 对比直接解原模型的 Original Formulation
print("               *** 直接解模型的Original Formulation ***               \n") 
# 构建模型
m = Model("SingleMachine")
m.setParam("OutputFlag", 1)
# 变量
x = [[0 for i in range(T)] for j in range(N)]
for jj in range(N):
    for tt in range(T-p[jj]+1):
        x[jj][tt] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='x[%d][%d]'%(jj,tt))

# 目标
obj = LinExpr()
for jj in range(N):
    for tt in range(T-p[jj]+1):
        obj += c[jj][tt]*x[jj][tt]
m.setObjective(obj,GRB.MINIMIZE)
# 约束
for jj in range(N):
    sum1 = LinExpr()
    for tt in range(T-p[jj]+1):
        sum1 += x[jj][tt]
    m.addConstr(sum1 == 1,'Constraint(1)_j=%d'%(jj))
for tt in range(T):
    sum1 = LinExpr()
    for jj in range(N):
        for ss in range( tt-p[jj]+1,tt+1 ): # 求和下限不从1开始，则上限+1
            sum1 += x[jj][ss]
    m.addConstr(sum1 <= 1,'Constraint(2)_t=%d'%(tt))
# 求解
m.update()
m.optimize()
print( 'Optimal value in Original Formulation: %g'%m.objVal )
# 输出最优解
#for jj in range(N):
#    for tt in range(T-p[jj]+1):
#        if ((x[jj][tt].x) > 0):
#            print( 'x[%d][%d] = %g'%(jj,tt,x[jj][tt].x) )
# 输出运行时间
print('直接解原模型,求解时间：%f'%(m.runtime))
