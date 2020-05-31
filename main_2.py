import numpy as np 
import geatpy as ea
from MyProblem_2 import MyProblem_2 #导入自定义问题接口
'''=========== 实例化问题对象==========='''
steps = [3,1,4,2,3,2,2,2,2,1]
problem = MyProblem_2() #生成问题对象

'''=========== 种群设置==========='''
NIND = 40 #种群规模
# 创建区域描述器，这里需要创建两个，分别是工序编码 机器编码
Encodings = ['RI', 'RI']
# crtfld (生成译码矩阵，俗称“区域描述器”)
Field1 = ea.crtfld(Encodings[0], problem.varTypes[:sum(steps)], problem.ranges[:,:sum(steps)], problem.borders[:,:sum(steps)]) # 创建区域描述器
Field2 = ea.crtfld(Encodings[1], problem.varTypes[sum(steps):], problem.ranges[:,sum(steps):], problem.borders[:,sum(steps):])
Fields = [Field1, Field2]
#实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
population = ea.PsyPopulation(Encodings, Fields, NIND)
'''=================算法参数设置==================='''
myAlgorithm = ea.moea_psy_NSGA2_templet(problem, population) # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 700    # 最大进化代数
myAlgorithm.drawing = 2   # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制过程动画）
"""===========================调用算法模板进行种群进化======================"""
NDSet = myAlgorithm.run()   # 执行算法模板，得到帕累托最优解集NDSet
NDSet.save()                # 把结果保存到文件中
# 输出
print('用时：%s 秒'%(myAlgorithm.passTime))
print('非支配个体数：%s 个'%(NDSet.sizes))
print('单位时间找到帕累托前沿点个数：%s 个'%(int(NDSet.sizes // myAlgorithm.passTime)))

